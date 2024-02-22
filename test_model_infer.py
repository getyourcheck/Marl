# %%
model_config = {"model_name":"actor",  "model_class":"actor",  "model_path":"facebook/opt-1.3b"}

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_config.get("model_path"))

input_text_list = ["A list of colors: red, blue", "Portugal is"]
inputs = tokenizer(input_text_list, return_tensors="pt", padding=True).to("cuda")
# inputs = { 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0]], device='cuda:0'),
#   'input_ids': tensor([[ 2, 250, 889, 9, 8089, 35, 1275, 6, 2440], [ 2, 22117, 39029, 16, 1, 1, 1, 1, 1]], device='cuda:0') }

# %%
from marl.model_backend import HfModelRunner
mr = HfModelRunner(model_config=model_config)

infer_config = {"max_new_tokens": 64}
output_inf = mr.infer(inputs.input_ids, step=1, **infer_config)
print(output_inf)

output_gen = mr.infer(inputs.input_ids, **infer_config)
print(output_gen)

# %%
import ray
from marl.model_backend import HfModelRunnerRayActor

ray.init()
mra = HfModelRunnerRayActor.remote(model_config)

infer_config = {"max_new_tokens": 64}
output_inf_r = ray.get(mra.infer.remote(inputs.input_ids, step=1, **infer_config))
print(output_inf_r)

output_gen_r = ray.get(mra.infer.remote(inputs.input_ids, **infer_config))
print(output_gen_r)

# %%
print("output_inf_r == output_inf: ", output_inf_r == output_inf)
print("output_gen_r == output_gen: ", output_gen_r == output_gen)

# %%
