# %%
from marl.coordinator import Coordinator
from marl.config.config import Config
from marl.config.config_consts import MODEL_TYPE_CRITIC, ENGINE_HUGGINGFACE

# INIT METHOD 1
# cluster_address = "auto"
# print(f"cluster_address={cluster_address}")
# model_configs_path = "projects/ppo/internlm2/1B/actor_critic_2gpu.py"
# model_configs = Config.from_file(model_configs_path)
# coordinator = Coordinator(cluster_address, model_configs)
# model_dict = coordinator.create_models()
# critic_model = model_dict["critic"]

# INIT METHOD 2
from marl.model_backend.hf_model_runner import HfModelRunner
critic_trainer_config = Config(
    dict(
        model_path="/cpfs01/shared/public/llm_model/ckpt/Luyou_1B/R-Luyou-1B-8k-D20240130-v1-hf/",
        model_type=MODEL_TYPE_CRITIC,
        trainer_type=ENGINE_HUGGINGFACE,
        train_kwargs=dict(
            micro_bsz=1,
            lr=1e-6,
            total_steps=1e9,
            lr_decay_rate=1,
            loss_type="per_token",
        ),
        parallel=dict(
            data=dict(size=1, mode="ddp"),
            tensor=dict(size=1, mode="1d"),
            pipeline=dict(size=1, interleaved_overlap=False),
            sequence=False,
        ),
    ),
)
critic_model = HfModelRunner(model_config=critic_trainer_config)
critic_model.initialize()

# %%
import torch
text_batch = ["I love Pixar.", "I don't care for Pixar."]
encoding = critic_model.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
labels = torch.tensor([1,0]).unsqueeze(0)

fwd_output = critic_model.infer(input_ids)
print("critic model forward output:", fwd_output.logits)
train_loss = critic_model.train(input_ids, labels, attention_mask)
print("critic model training loss:", train_loss)