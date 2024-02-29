# %%
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
# from accelerate import Accelerator
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

output_inf = actor_model.infer(input_ids, step=1)
print("[INFO] 1. One-step inference:", output_inf)

generate_kwargs = {"max_new_tokens": 64}
output_gen_ref = actor_model.infer_async(input_ids, generate_kwargs=generate_kwargs)
print("[INFO] 2.1 generation async:", output_gen_ref)
output_gen_res = actor_model.infer_get(output_gen_ref)
print("[INFO] 2.2 generation result:", output_gen_res)

output_gen = actor_model.infer(input_ids, generate_kwargs=generate_kwargs)
print("[INFO] 3. generation sync:", output_gen)

print("[INFO] output_gen_res == output_gen:", output_gen_res == output_gen)
print("[INFO] output in str:")
for i, outstr in enumerate(output_gen.output_str):
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

# %% 5. clean up
coordinator.clean_up()
