# %%
from marl.coordinator import Coordinator
from marl.config import Config

# INIT METHOD 1
cluster_address = "auto"
print(f"cluster_address={cluster_address}")
model_configs_path = "projects/ppo/internlm2/1B/actor_reward_2gpu.py"
model_configs = Config.from_file(model_configs_path)
coordinator = Coordinator(cluster_address, model_configs)
model_dict = coordinator.create_models()
reward_model = model_dict["reward"]

# INIT METHOD 2
# from marl.model_backend.hf_model_runner import HfModelRunner
# reward_trainer_config = Config(
#     dict(
#         model_path="/cpfs01/shared/public/llm_model/ckpt/Luyou_1B/R-Luyou-1B-8k-D20240130-v1-hf/",
#         model_type="reward",
#         trainer_type="huggingface",
#         parallel=dict(
#             data=dict(size=1, mode="ddp"),
#             tensor=dict(size=1, mode="1d"),
#             pipeline=dict(size=1, interleaved_overlap=False),
#             sequence=False,
#         ),
#     ),
# )
# reward_model = HfModelRunner(model_config=reward_trainer_config)
# reward_model.initialize()

# %%
print()
chat_1 = [
    {"role": "user", "content": "Hello! What's your name?"},
    {"role": "assistant", "content": "My name is InternLM2!"},
]
chat_2 = [
    {"role": "user", "content": "Hello! What's your name?"},
    {"role": "assistant", "content": "I don't know."},
]

chat_1_str = "<s><|im_start|>user\nHello! What's your name?<|im_end|>\n<|im_start|>assistant\nMy name is InternLM2!<|im_end|>\n"
score0 = reward_model.infer(inputs=chat_1_str)  # get_score
print("score0:", score0.logits)

score1 = reward_model.infer(inputs=chat_1)  # get_score
print("score1:", score1.logits)
assert score0.logits[0] == score1.logits[0]

score2 = reward_model.infer(inputs=chat_2)  # get_score
print("score2:", score2.logits)
assert score1.logits > score2.logits

scores = reward_model.infer(inputs=[chat_1, chat_2])  # get_scores
print("scores:", scores.logits)


# %%
