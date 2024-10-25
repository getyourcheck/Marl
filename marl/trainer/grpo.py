from loguru import logger

from ..loss import PPOPolicyLoss, PretrainLoss, GRPOPolicyLoss
from ..model_server.base_model_server import BaseModelServer
from ..timer import Timer


class GRPOTrainer:
    def __init__(
        self,
        policy_model: BaseModelServer,
        policy_micro_bs=2,
        policy_learn_time=1,
        policy_minibatch=None,
        ppo_loss_weight=1.0,
        pretrain_loss_weight=0.5,
        pretrain_criterion=PretrainLoss(label_smoothing=0),
        policy_criterion=GRPOPolicyLoss(cliprange=0.2),
        use_varlen_attn=False,
        **kwargs,
    ):
        # policy
        self.policy_model = policy_model
        self.policy_learn_time = policy_learn_time
        self.policy_minibatch = policy_minibatch
        self.policy_micro_bs = policy_micro_bs

        self.ppo_loss_weight = ppo_loss_weight
        self.pretrain_loss_weight = pretrain_loss_weight
        self.pretrain_criterion = pretrain_criterion
        self.policy_criterion = policy_criterion

        self.use_varlen_attn = use_varlen_attn

    def policy_learn(self, trajectories, pretrain_data=None):
        if self.policy_minibatch is None:
            self.policy_minibatch = len(trajectories.output_ids)
        assert len(trajectories.output_ids) % self.policy_minibatch == 0
        policy_updates = len(trajectories.output_ids) // self.policy_minibatch
        ppo_loss = []
        pretrain_loss = []

        for _ in range(self.policy_learn_time):
            for i in range(policy_updates):
                logger.info(
                    "[Policy Train] start policy trains {}/{} | {}".format(
                        i + 1, policy_updates, _ + 1
                    )
                )
                # prompt train data
                begin = i * self.policy_minibatch
                end = begin + self.policy_minibatch

                train_input_ids = [trajectories.output_ids[begin:end, :]]
                train_attention_mask = [trajectories.attention_mask[begin:end, :]]
                train_criterion = [self.policy_criterion]
                loss_weights = [self.ppo_loss_weight]
                micro_batch_size = [self.policy_micro_bs]

                train_lables = [
                    dict(
                        input_ids=trajectories.output_ids[begin:end, :],
                        old_logprobs=trajectories.policy_logprobs[begin:end, :],
                        ref_logprobs=trajectories.ref_logprobs[begin:end, :],
                        advantages=trajectories.advantages[begin:end, :],
                        mask=trajectories.action_mask[begin:end, :],
                    ),
                ]
                train_position_ids = [None]
                cumulative_len = [None]
                max_seqlen = [None]
                # pretrain data
                if pretrain_data is not None:
                    logger.info(
                        "[Policy Train] pretrain data "
                        f'{pretrain_data["input_ids"].shape}'
                    )
                    train_input_ids.append(pretrain_data["input_ids"])
                    train_lables.append(pretrain_data["labels"])
                    train_position_ids.append(pretrain_data["position_ids"])
                    train_attention_mask.append(None)
                    train_criterion.append(self.pretrain_criterion)
                    loss_weights.append(self.pretrain_loss_weight)
                    micro_batch_size.append(self.policy_micro_bs)
                    cumulative_len.append(pretrain_data["cumulative_len"])
                    max_seqlen.append(pretrain_data["max_seqlen"])

                with Timer("policy_model.train"):
                    p_loss = self.policy_model.train(
                        input_ids=train_input_ids,
                        labels=train_lables,
                        attention_mask=train_attention_mask,
                        position_ids=train_position_ids,
                        criterion=train_criterion,
                        loss_weights=loss_weights,
                        micro_batch_size=micro_batch_size,
                        cumulative_len=cumulative_len,
                        max_seqlen=max_seqlen,
                        use_varlen_attn=self.use_varlen_attn,
                    )
                if isinstance(p_loss, list):
                    ppo_loss.append(p_loss[0].item())
                    pretrain_loss.append(p_loss[1].item())
                    logger.info(
                        f"[Policy Train] prompt data: {train_input_ids[0].shape}, ppo loss: {p_loss[0].item()}; pretrain data: {train_input_ids[1].shape}, pretrain loss: {p_loss[1].item()}"  # noqa: E501
                    )
                else:
                    ppo_loss.append(p_loss.item())
                    logger.info(
                        f"[Policy Train] prompt data: {train_input_ids[0].shape}, ppo loss: {p_loss.item()}"  # noqa: E501
                    )

        with Timer("policy_model.sync_model"):
            self.policy_model.sync_model()
        return ppo_loss, pretrain_loss
