# %%

import torch
from marl.config import Config
from marl.tokenizer.tokenizer_utils import get_tokenizer

trainer_config = Config(
    dict(
        model_path="internlm/internlm2-chat-1_8b-sft",
        torch_dtype=torch.float16,
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

tokenizer_path = trainer_config.get("tokenizer_path", trainer_config.get("model_path"))
tokenizer = get_tokenizer(tokenizer_path, trust_remote_code=True)

input_text_list = ["A list of colors: red, blue", "Portugal is"]
inputs = tokenizer(input_text_list, return_tensors="pt", padding=True).to("cuda")
# inputs = { 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0]], device='cuda:0'),
#   'input_ids': tensor([[ 2, 250, 889, 9, 8089, 35, 1275, 6, 2440], [ 2, 22117, 39029, 16, 1, 1, 1, 1, 1]], device='cuda:0') }

# %%
from marl.model_backend.hf_model_runner import HfModelRunner

mr = HfModelRunner(model_config=trainer_config)
mr.initialize()

# %%
output_inf = mr.infer(inputs.input_ids)  # by default: step=1, output_logits=True
print(output_inf)

output_gen = mr.generate(inputs.input_ids, step=1, output_logits=True)
print(output_gen)

print("[ASSERT TRUE] output_gen first token's logits == output_inf last token's result")
assert torch.equal(output_gen.logits[0], output_inf.logits[:, -1, :])

# %%
import ray
from marl.model_backend.hf_model_runner import HfModelRunnerRayActorGroup

ray.init(ignore_reinit_error=True)
mra = HfModelRunnerRayActorGroup(name="mra", config=trainer_config)

output_inf_ray = mra.infer(inputs.input_ids)  # by default: output_logits=True
print(output_inf_ray)
print("[ASSERT TRUE] output_inf_ray == output_inf: ", output_inf_ray == output_inf, "\n")
assert output_inf_ray == output_inf

output_gen_ray = mra.generate(inputs.input_ids, step=1, output_logits=True)
print(output_gen_ray)
print("[ASSERT TRUE] output_gen_ray == output_gen: ", output_gen_ray == output_gen, "\n")
assert output_gen_ray == output_gen

# %%