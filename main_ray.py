# %%
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
# from accelerate import Accelerator
from marl.coordinator import Coordinator
from marl.config import Config

# %% 1. load a pretrained model
model_configs = [ Config(x) for x in [
    {"model_name":"actor",  "model_class":"actor",  "model_path":"facebook/opt-1.3b"},
    {"model_name":"critic", "model_class":"critic", "model_path":"facebook/opt-1.3b"},
    {"model_name":"ref",    "model_class":"ref",    "model_path":"facebook/opt-1.3b"},
    {"model_name":"reward", "model_class":"reward", "model_path":"facebook/opt-1.3b"},
]]

coordinator = Coordinator(model_configs=model_configs)
models = coordinator.create_models()
models_dict = dict(zip([ mc.model_name for mc in model_configs], models))
tokenizer = models_dict['actor'].tokenizer

# %% 2. prepare models
print()
actor_model = models_dict['actor']
actor_model_model_ref = actor_model.model_ref
print("[INFO] Reference of actor_model's model:", actor_model_model_ref)
print("[INFO] Force pulling the model to local ...")
print(actor_model.model_get())

# %% 3. infer a query
print()
input_text_list = ["A list of colors: red, blue", "Portugal is"]
inputs = tokenizer(input_text_list, return_tensors="pt", padding=True)

infer_config = {"max_new_tokens": 64}
output_inf = actor_model.infer(inputs.input_ids, step=1, **infer_config)
print("[INFO] 1. One-step inference:", output_inf)

output_gen_ref = actor_model.infer_async(inputs.input_ids, **infer_config)
print("[INFO] 2.1 generation async:", output_gen_ref)
output_gen_res = actor_model.infer_get(output_gen_ref)
print("[INFO] 2.2 generation result:", output_gen_res)

output_gen = actor_model.infer(inputs.input_ids, **infer_config)
print("[INFO] 3. generation sync:", output_gen)

print("[INFO] assert output_gen_res == output_gen:", output_gen_res == output_gen)
print("[INFO] output in str:")
for i, outstr in enumerate(output_gen.output_str):
    print(f"[INFO] {i}/n:\n{outstr}")

# %% 4. train
print()
print("[INFO] [WIP] actor_model.train() outputs loss:", actor_model.train(inputs.input_ids, inputs.input_ids))
