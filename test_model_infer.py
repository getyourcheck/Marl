# %%
from marl.config import Config

trainer_config = Config(
    dict(
        model_path="facebook/opt-1.3b",
        trainer_type="huggingface",
        parallel=dict(
            data=dict(size=1),
            tensor=dict(size=1, mode="1d"),
            pipeline=dict(size=1, interleaved_overlap=False),
        ),
    ),
)

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(trainer_config.get("model_path"))

input_text_list = ["A list of colors: red, blue", "Portugal is"]
inputs = tokenizer(input_text_list, return_tensors="pt", padding=True).to("cuda")
# inputs = { 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0]], device='cuda:0'),
#   'input_ids': tensor([[ 2, 250, 889, 9, 8089, 35, 1275, 6, 2440], [ 2, 22117, 39029, 16, 1, 1, 1, 1, 1]], device='cuda:0') }

# %%
from marl.model_backend.hf_model_runner import HfModelRunner

mr = HfModelRunner(model_config=trainer_config)
mr.initialize()

infer_config = {"max_new_tokens": 64}
output_inf = mr.infer(inputs.input_ids, step=1, **infer_config)
print(output_inf)

output_gen = mr.infer(inputs.input_ids, **infer_config)
print(output_gen)

# %%
import ray
from marl.model_backend.hf_model_runner import HfModelRunnerRayActorGroup

ray.init(ignore_reinit_error=True)
mra = HfModelRunnerRayActorGroup(name="mra", config=trainer_config)

infer_config = {"max_new_tokens": 64}
output_inf_ray = mra.infer(inputs.input_ids, step=1, **infer_config)
print(output_inf_ray)
print("output_inf_ray == output_inf: ", output_inf_ray == output_inf, "\n")
assert output_inf_ray == output_inf

output_gen_ray = mra.infer(inputs.input_ids, **infer_config)
print(output_gen_ray)
print("output_gen_ray == output_gen: ", output_gen_ray == output_gen, "\n")
assert output_gen_ray == output_gen
