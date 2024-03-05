# %%
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
# from accelerate import Accelerator
import torch
from marl.coordinator import Coordinator
from marl.config import Config

# %% 1. load a pretrained model
cluster_address = "auto"
print(f"cluster_address={cluster_address}")
model_configs_path = "projects/ppo/internlm2/1B/actor_reward_2gpu.py"
model_configs = Config.from_file(model_configs_path)
coordinator = Coordinator(cluster_address, model_configs)
model_dict = coordinator.create_models()

# %% 2. prepare models
print()
actor_model = model_dict["actor"]
actor_model_model_ref = actor_model.model_ref
print("[INFO] Reference of actor_model's model:", actor_model_model_ref)
print("[INFO] Force pulling the model to local ...")
print(actor_model.model_get())

# %% 3. infer a query
print()
input_text_list = ["A list of colors: red, blue", "Portugal is"]
inputs = actor_model.tokenizer(input_text_list, return_tensors="pt", padding=True)
inputs.input_ids = inputs.input_ids.to("cuda")
input_ids = inputs.input_ids

output_inf = actor_model.infer(input_ids)  # by default: output_logits=True
print("[INFO] 1. one-step inference:", output_inf)

output_inf_ref = actor_model.infer_async(input_ids)
print("[INFO] 2.1 one-step inference async:", output_inf_ref)
output_inf_ray = actor_model.infer_get(output_inf_ref)
print("[INFO] 2.2 one-step inference result:", output_inf_ray)
print("[INFO] ASSERT output_inf_ray == output_inf:", output_inf_ray == output_inf)
assert output_inf_ray == output_inf

output_gen_ref = actor_model.generate_async(
    input_ids, step=1, output_logits=True, output_str=True
)
output_gen_ray = actor_model.generate_get(output_gen_ref)
print("[INFO] 3. generation async:", output_gen_ray)
print("[INFO] ASSERT output_gen first token's logits == output_inf last token's result")
assert torch.equal(output_gen_ray.logits[0], output_inf.logits[:, -1, :])

print("[INFO] output in str:")
for i, outstr in enumerate(output_gen_ray.output_str):
    print(f"[INFO] {i}/n:\n{outstr}")

# %% 4. train
print()
# STEP_INVERTAL=2 blocks optimizer.step() and enables reproducibility
actor_model.set_seed()
train_loss = actor_model.train(input_ids, input_ids, step_interval=2)
print("[INFO] 4.1 training outputs loss:", train_loss)

actor_model.set_seed()
train_loss_ref = actor_model.train_async(input_ids, input_ids, step_interval=2)
print("[INFO] 5.1 training async:", train_loss_ref)
train_loss_res = actor_model.train_get(train_loss_ref)
print("[INFO] 5.2 training result:", train_loss_res)
print("[INFO] 4.1 training loss == 5.2 training loss:", train_loss == train_loss_res)

# %% 5. reward models
print()
reward_model = model_dict["reward"]
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
print("[INFO] 6.1 score0:", score0.logits)

score1 = reward_model.infer(inputs=chat_1)  # get_score
print("[INFO] 6.2 score1:", score1.logits)

score2 = reward_model.infer(inputs=chat_2)  # get_score
print("[INFO] 6.3 score2:", score2.logits)
assert score1.logits > score2.logits
print("[INFO] 6.1 score1 > 6.2 score2:", score1.logits > score2.logits)

scores = reward_model.infer(inputs=[chat_1, chat_2])  # get_scores
print("[INFO] 6.4 scores:", scores.logits)
# %% 6. clean up
coordinator.clean_up()
