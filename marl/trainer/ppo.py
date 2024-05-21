import time
import torch
from loguru import logger
from marl.model_server.base_model_server import BaseModelServer

import sys
sys.path.extend(["./", "marl/dataset"])
from marl.loss.actor_loss import ActorLoss
from marl.loss.critic_loss import CriticLoss
from marl.loss.pretrain_loss import PretrainLoss
from marl.timer import Timer


class PPOTrainer(object):
    def __init__(
            self, 
            policy_model, 
            value_model, 
            actor_micro_bs=2,
            critic_micro_bs=2,
            policy_learn_time=1,
            value_learn_time=1,
            ppo_minibatch=512,
            value_minibatch=512,
            pt_minibatch=32,
            train_minibatch=None,
            pt_criterion = PretrainLoss(loss_factor=1.0),
            policy_criterion = ActorLoss(cliprange=0.2, loss_type="per_seq"),
            value_criterion = CriticLoss(cliprange_value=0.5, loss_type="per_seq"),
            **kwargs,
        ):
        
        self.ppo_minibatch = ppo_minibatch
        self.value_minibatch = value_minibatch
        self.actor_micro_bs = actor_micro_bs
        self.critic_micro_bs = critic_micro_bs
        # policy
        self.policy_model = policy_model
        self.policy_learn_time = policy_learn_time
        self.pt_minibatch = pt_minibatch
        self.train_minibatch = train_minibatch
        self.policy_minibatch = ppo_minibatch

        # value
        self.value_model = value_model
        self.value_learn_time = value_learn_time
        self.value_minibatch = value_minibatch

        self.pt_criterion = pt_criterion
        self.policy_criterion = policy_criterion
        self.value_criterion = value_criterion

    def policy_learn(self, trajectories, policy_model:BaseModelServer):
        policy_updates = len(trajectories.output_ids) // self.policy_minibatch
        policy_loss = []
        # TODO, 
        pt_loss = []
        # if self.pt_minibatch is not None:
        #     assert trajectories.pt_data is not None, "Make sure pretrain data in your data loader!!!"

        for _ in range(self.policy_learn_time):
            for i in range(policy_updates):
                logger.info('[Policy Train] start policy trains {}/{} | {}'.format(i + 1, policy_updates, _ + 1))
                begin = i * self.policy_minibatch
                end = begin + self.policy_minibatch
                policy_batch_inputs = {
                    "input_ids": trajectories.output_ids[begin:end, :],
                    "policy_logprobs": trajectories.policy_logprobs[begin:end, :],
                    "advs": trajectories.advantages[begin:end, :],
                    "action_mask": trajectories.action_mask[begin:end, :],
                    "attention_mask": trajectories.attention_mask[begin:end, :]
                }
                assert len(policy_batch_inputs['input_ids']) == self.policy_minibatch, "[Policy learn] make sure len(policy_batch_inputs) == self.policy_minibatch"

                loss_factor = 1.0
                labels = dict(input_ids=policy_batch_inputs["input_ids"],
                            old_logprobs=policy_batch_inputs["policy_logprobs"],
                            advantages=policy_batch_inputs["advs"],
                            mask=policy_batch_inputs["action_mask"],
                            loss_factor=torch.tensor(loss_factor),
                        )
                train_input_ids = [policy_batch_inputs["input_ids"], ]
                train_lables = [labels, ]
                train_attention_mask = [policy_batch_inputs["attention_mask"], ]
                train_criterion = [self.policy_criterion, ]
                loss_weights=[1.0, ]
                micro_batch_size=[self.actor_micro_bs, ]
                # pretrain data
                if self.pt_minibatch is not None:
                    logger.info(f'[Policy Train] policy train with pretrain data {trajectories.pretrain_input_ids.shape}')
                    train_input_ids.append(trajectories.pretrain_input_ids)
                    train_lables.append(trajectories.pretrain_labels)
                    train_attention_mask.append(None)
                    train_criterion.append(None)
                    loss_weights.append(0.5)
                    micro_batch_size.append(1)
                # for k, v in labels.items():
                #     print("[Policy Train]]", k, v.shape)
                s_t = time.time()
                p_loss = policy_model.train(
                    input_ids=train_input_ids,
                    labels=train_lables,     
                    attention_mask=train_attention_mask,
                    criterion=train_criterion,
                    loss_weights=loss_weights,
                    micro_batch_size=micro_batch_size
                )
                if self.pt_minibatch is not None:
                    policy_loss.append(p_loss[0].item())
                    pt_loss.append(p_loss[1].item())
                    logger.info(f"[actor train] duration: {round(time.time() - s_t, 2)} s, prompt: {self.policy_minibatch} batch, Policy loss: {p_loss[0].item()}; pretrain: {self.pt_minibatch} batch, Pretrain loss: {p_loss[1].item()}")
                else:
                    policy_loss.append(p_loss.item())
                    logger.info(f"[actor train] duration: {round(time.time() - s_t, 2)} s, {self.policy_minibatch} batch, Policy loss: {p_loss.item()}")

        with Timer("policy_model.sync_model") as t:
            policy_model.sync_model()
        return policy_loss, pt_loss

    def value_learn_async(self, trajectories, value_model:BaseModelServer):
        value_updates = len(trajectories.output_ids) // self.value_minibatch
        value_loss = []
        assert value_updates == 1 and self.policy_learn_time == 1, f"value_updates={value_updates} * self.policy_learn_time={self.policy_learn_time} > 1"
        s_t = time.time()
        value_batch_inputs, labels = self._value_learn_prepare(0, 0, trajectories, value_updates)
        v_loss_ref = value_model.train_async(
            input_ids=value_batch_inputs["input_ids"],
            labels=labels,     
            attention_mask=value_batch_inputs["attention_mask"],
            criterion=self.value_criterion,
            micro_batch_size=self.critic_micro_bs,
        )
        logger.info(f"[critic train] async duration: {round(time.time() - s_t, 2)} s, {self.value_minibatch} batch")
        value_loss.append(v_loss_ref)
        return value_loss
    
    def value_learn_get(self, value_loss_ref, value_model:BaseModelServer):
        with Timer("value_model.train_get"):
            return [value_model.train_get(ref).item() for ref in value_loss_ref]

    def value_learn(self, trajectories, value_model:BaseModelServer):
        value_updates = len(trajectories.output_ids) // self.value_minibatch
        value_loss = []

        for learn_i in range(self.policy_learn_time):
            for step_i in range(value_updates):
                s_t = time.time()
                value_batch_inputs, labels = self._value_learn_prepare(step_i, learn_i, trajectories, value_updates)
                v_loss = value_model.train(
                    input_ids=value_batch_inputs["input_ids"],
                    labels=labels,     
                    attention_mask=value_batch_inputs["attention_mask"],
                    criterion=self.value_criterion,
                    micro_batch_size=self.critic_micro_bs,
                )
                logger.info(f"[critic train] duration: {round(time.time() - s_t, 2)} s, {self.value_minibatch} batch,value loss: {v_loss.item()}")
                value_loss.append(v_loss.item())
        return value_loss

    def _value_learn_prepare(self, step_i, learn_i, trajectories, value_updates):
        logger.info('[Value Train] start value trains {}/{} | {}'.format(step_i + 1, value_updates, learn_i + 1))
        begin = step_i * self.value_minibatch
        end = begin + self.value_minibatch
        value_batch_inputs = {
            "input_ids": trajectories.output_ids[begin:end, :],
            "old_values": trajectories.old_values[begin:end, :],
            "returns": trajectories.returns[begin:end, :],
            "action_mask": trajectories.action_mask[begin:end, :],
            "attention_mask": trajectories.attention_mask[begin:end, :]
        }
        assert len(value_batch_inputs['input_ids']) == self.value_minibatch, "[Value learn] make sure len(value_batch_inputs) == self.value_minibatch"

        loss_factor = 1.0
        labels = dict(
            old_values=value_batch_inputs["old_values"],
            returns=value_batch_inputs["returns"],
            mask=value_batch_inputs["action_mask"],
            loss_factor=torch.tensor(loss_factor),
        )
        return value_batch_inputs, labels
