# %%

import torch
from marl.config.config import Config
from marl.tokenizer.tokenizer_utils import get_tokenizer
from marl.ppo_learner_utils import calc_kl_penalty
from marl.policy_output import logprobs_from_logits

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
output_inf = mr.infer(inputs.input_ids, output_logprobs=True)  # by default: step=1, output_logits=True
print(output_inf)

output_gen = mr.generate(inputs.input_ids, step=1, output_logits=True)
print(output_gen)

print("[ASSERT TRUE] output_gen first token's logits == output_inf last token's result")
assert torch.equal(output_gen.logits[0], output_inf.logits[:, -1, :])

logprob = output_inf["logprobs"]
ref_logprob = logprobs_from_logits(logits=output_inf["logits"][:, :-1, :], labels=inputs.input_ids[:, 1:])
kl = calc_kl_penalty(logprob, ref_logprob)
print(f"[ASSERT ZERO] kl divergence of logprob and ref_logprob: {kl}")
assert all(kl==0)

# %%
import ray
from marl.model_backend.hf_model_runner import HfModelRunnerRayActorGroup

ray.init(ignore_reinit_error=True)
mra = HfModelRunnerRayActorGroup(name="mra", config=trainer_config)

output_inf_ray = mra.infer(inputs.input_ids, output_logprobs=True)  # by default: output_logits=True
print(output_inf_ray)
print("[ASSERT TRUE] output_inf_ray == output_inf: ", output_inf_ray == output_inf, "\n")
assert output_inf_ray == output_inf

output_gen_ray = mra.generate(inputs.input_ids, step=1, output_logits=True)
print(output_gen_ray)
print("[ASSERT TRUE] output_gen_ray == output_gen: ", output_gen_ray == output_gen, "\n")
assert output_gen_ray == output_gen

logprob = output_inf_ray["logprobs"]
ref_logprob = logprobs_from_logits(logits=output_inf_ray["logits"][:, :-1, :], labels=inputs.input_ids[:, 1:])
kl = calc_kl_penalty(logprob, ref_logprob)
print(f"[ASSERT ZERO] kl divergence of logprob and ref_logprob: {kl}")
assert all(kl==0)
# %%