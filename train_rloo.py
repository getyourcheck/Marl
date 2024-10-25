import argparse
import json
import os
import shutil
import time

from loguru import logger

from marl.config import Config
from marl.coordinator import Coordinator
from marl.dataset import MessageIter
from marl.envs import TxtEnv
from marl.repeaters.loo import LOORepeater
from marl.timer import Timer
from marl.trainer.rloo import RLOOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM")
    parser.add_argument(
        "-c",
        "--config",
        help="config file name or path.",
        type=str,
        default="projects/internlm2/1B/ppo_ds_vllm_8gpu.py",
    )
    parser.add_argument(
        "-w",
        "--work_dir",
        help="the dir to save logs and models",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-a", "--address", help="ray head address", type=str, default="auto"
    )
    args = parser.parse_args()
    return args


def validate_config(config: Config):
    assert config["model_configs"] is not None
    assert config["model_configs"]["policy"] is not None
    assert config["model_configs"]["policy"]["model_path"] is not None
    assert config["dataset_config"] is not None
    assert config["rollout_config"] is not None
    assert config["rollout_config"]["generate_kwargs"] is not None
    assert config["rollout_config"]["max_new_tokens"] is not None


if __name__ == "__main__":
    args = parse_args()
    assert args.config is not None, "config should not be None"
    work_dir = args.work_dir
    if work_dir is None:
        work_dir = os.getcwd() + "/trainlog_" + time.strftime("%Y-%m-%d-%H:%M:%S")
    work_dir = os.path.abspath(work_dir)
    logger.info(f"using work_dir: {work_dir}")
    os.makedirs(work_dir, exist_ok=True)
    # save original config
    shutil.copy2(args.config, f"{work_dir}/{os.path.basename(args.config)}")

    logger.add(
        f"{work_dir}/train_rlhf.log",
        filter=lambda record: record["extra"].get("name") == "train",
    )
    logger_train = logger.bind(name="train")

    config = Config.from_file(args.config)
    logger.info("#################### CONFIG BGN ####################")
    for k, v in config.items():
        logger.info(f"{k}: {v}")
    logger.info("#################### CONFIG END ####################")

    # init model
    cluster_address = args.address
    if cluster_address != "auto":
        cluster_address = f"ray://{cluster_address}:10001"
    logger.info(f"cluster_address={cluster_address}")
    coordinator = Coordinator(cluster_address, config)
    model_dict = coordinator.create_models()
    ref_model = model_dict["reference"]
    policy_model = model_dict["policy"]
    reward_model = model_dict["reward"]

    # init prompt & pretrain dataset
    prompt_dataset_config = config["prompt_dataset_config"]
    prompt_mes_iter = MessageIter(
        tokenizer=ref_model.tokenizer, **prompt_dataset_config
    )
    pretrain_dataset_config = config.get("pretrain_dataset_config", {})
    # pretrain_mes_iter = MessageIter(
    #     tokenizer=ref_model.tokenizer, **pretrain_dataset_config)
    from marl.dataset.intern_repo import get_pretrain_data

    pretrain_data_iter = get_pretrain_data(**pretrain_dataset_config)

    # init txt env
    rollout_config = config.get("rollout_config", {})
    txt_env = TxtEnv(
        policy_model=policy_model,
        reward_model=reward_model,
        prompt_mes_iter=prompt_mes_iter,
        pretrain_mes_iter=None,  # None
        **rollout_config,
    )
    # init repeater
    repeater_config = config.get("repeater_config", {})
    ppo_repeater = LOORepeater(
        ref_model=ref_model,
        policy_model=policy_model,
        env=txt_env,
        **repeater_config,
    )
    # init trainer
    train_config = config.get("train_config", {})
    ppo = RLOOTrainer(policy_model=policy_model, **train_config)
    save_interval = train_config["save_interval"]
    max_train_step = train_config.get("max_train_step", float("inf"))
    resume_step = train_config.get("resume_step", -1)
    txt_env.resume_step = resume_step
    async_learn = train_config.get("async_learn", True)

    step = max(0, resume_step)
    while step <= max_train_step:
        s_t = time.time()
        with Timer(f"step {step}: end_to_end"):
            # generate trajectories
            trajectories = txt_env.rollout(display=True)

            # deal with trajectories
            trajectories = ppo_repeater.process(trajectories)

            # policy learn
            ppo_loss, pt_loss = None, None
            if resume_step > 0:
                for _ in range(resume_step, 0):
                    pretrain_data = (
                        next(pretrain_data_iter)
                        if pretrain_data_iter is not None
                        else None
                    )
                resume_step = -1
            pretrain_data = (
                next(pretrain_data_iter) if pretrain_data_iter is not None else None
            )
            ppo_loss, pt_loss = ppo.policy_learn(
                trajectories, pretrain_data=pretrain_data
            )
            logger_train.info(
                f"[Policy Train] Step: {step}, "
                f"ppo loss: {ppo_loss}, pretrain loss: {pt_loss}"
            )

        logger_train.info(f"rewards: {trajectories.rewards.mean()}")
        if config["rollout_config"].get("write_to_file", True):
            if not os.path.exists(f"{work_dir}/rollouts"):
                os.makedirs(f"{work_dir}/rollouts")
            with open(f"{work_dir}/rollouts/step{step}_rollout.log", "a") as file:
                for output_s, r, a, c in zip(
                    trajectories.output_str,
                    trajectories.rewards,
                    trajectories.rloo_advantages,
                    trajectories.clipped_rewards,
                ):
                    file.write(
                        output_s
                        + "\n"
                        + "Reward: "
                        + str(r.item())
                        + "\n"
                        + "Clipped Reward: "
                        + str(c.item())
                        + "\n"
                        + "RLOO Advantages: "
                        + str(a.item())
                        + "\n"
                        + "=" * 30
                        + "\n"
                    )
        summaries = dict(
            reward_mean=trajectories.rewards.mean().item(),
            reward_std=trajectories.rewards.std().item(),
            rloo_advantages_mean=trajectories.rloo_advantages.mean().item(),
            new_tokens_mean=trajectories.action_mask.sum(-1).float().mean().item(),
            new_tokens_std=trajectories.action_mask.sum(-1).float().std().item(),
            kl=trajectories.kl.mean().item(),
            seq_kl=trajectories.seq_kl.mean().item(),
            entropy=trajectories.entropy.mean().item(),
            step=step,
            policy_loss=ppo_loss,
            pretrain_loss=pt_loss,
        )
        with open(f"{work_dir}/train_rlhf.log.jsonl", "a") as f:
            f.write(json.dumps(summaries) + "\n")
        logger_train.info(f"[end to end] duration: {time.time() - s_t} s")

        step += 1
        if (step % save_interval == 0) or (step == max_train_step):
            policy_model.save(f"{work_dir}/ckpt/policy_model/{step}")
