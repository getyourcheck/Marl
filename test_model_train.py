# %%
import torch
from marl.config import Config
from marl.tokenizer.tokenizer_utils import get_tokenizer

trainer_config = Config(
    dict(
        model_path="internlm/internlm2-chat-1_8b-sft",
        torch_dtype=torch.float16,
        trainer_type="huggingface",
        train_kwargs=dict(
            micro_bsz=1,
            lr=1e-6,
            total_steps=1e9,
            lr_decay_rate=1,
            loss_type="per_token",
        ),
        parallel=dict(
            data=dict(size=1),
            tensor=dict(size=1, mode="1d"),
            pipeline=dict(size=1, interleaved_overlap=False),
        ),
    ),
)

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # avoid Accelerator() from using multi-GPU

tokenizer_path = trainer_config.get("tokenizer_path", trainer_config.get("model_path"))
tokenizer = get_tokenizer(tokenizer_path, trust_remote_code=True)

input_text_list = ["A list of colors: red, blue", "Portugal is"]
inputs = tokenizer(input_text_list, return_tensors="pt", padding=True).to("cuda")
# inputs = { 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0]], device='cuda:0'),
#   'input_ids': tensor([[ 2, 250, 889, 9, 8089, 35, 1275, 6, 2440], [ 2, 22117, 39029, 16, 1, 1, 1, 1, 1]], device='cuda:0') }

# %%
from marl.utils import set_seed
from marl.model_backend.hf_model_runner import HfModelRunner
from torch.nn import CrossEntropyLoss

mr = HfModelRunner(model_config=trainer_config)
mr.initialize()

# STEP_INVERTAL=999 blocks optimizer.step() and enables reproducibility
input_ids = inputs.input_ids  # shape: [2, 9]

# < criterion test>
set_seed(1234)
loss1a = mr.train(input_ids, input_ids, step_interval=999)
print("[main] loss1a:", loss1a)

set_seed(1234)
loss1b = mr.train(input_ids, input_ids, criterion=CrossEntropyLoss, step_interval=999)
print("[main] loss1b:", loss1b)

print("[main]", "loss1a == loss1b: ", loss1a == loss1b, "\n")
assert loss1a == loss1b
# </criterion test>

# < multiple losses test>
set_seed(1234)
input_ids_0 = input_ids[0].unsqueeze(dim=0)  # shape: [1, 9]
input_ids_1 = input_ids[1].unsqueeze(dim=0)  # shape: [1, 9]
loss1c = mr.train(
    input_ids=[input_ids_0, input_ids_1],
    labels=[input_ids_0, input_ids_1],
    attention_mask=[None, None],
    criterion=[None, None],
    loss_weights=[1.0, 1.0],
    step_interval=999,
)
print("[main] loss1c:", loss1c)

set_seed(1234)
loss1d1 = mr.train(input_ids_0, input_ids_0, step_interval=999)
loss1d2 = mr.train(input_ids_1, input_ids_1, step_interval=999)
loss1d = (loss1d1 * 1.0 + loss1d2 * 1.0) / 2
print("[main] loss1d:", loss1d)

print("[main]", "loss1c == loss1d: ", loss1c == loss1d, "\n")
assert loss1c == loss1d
# </multiple losses test>

# %%
import ray
from marl.model_backend.hf_model_runner import HfModelRunnerRayActorGroup

ray.init(ignore_reinit_error=True)
input_ids = inputs.input_ids  # shape: [2, 9]

# < reproduce loss1a criterion test on ray>
mra = HfModelRunnerRayActorGroup("mra", trainer_config)
mra.set_seed(1234)
loss1a_ray = mra.train(input_ids, input_ids, step_interval=999)
print("[main] loss1a_ray", loss1a_ray)
print("[main]", "loss1a_ray == loss1a", loss1a_ray == loss1a, "\n")
assert loss1a_ray == loss1a
# </reproduce loss1a criterion test on ray>

# < reproduce loss1c multiple loss test on ray>
mra.set_seed(1234)
loss1c_ray = mra.train(
    input_ids=[input_ids_0, input_ids_1],
    labels=[input_ids_0, input_ids_1],
    attention_mask=[None, None],
    criterion=[None, None],
    loss_weights=[1.0, 1.0],
    step_interval=999,
)
print("[main] loss1c_ray", loss1a_ray)
print("[main]", "loss1c_ray == loss1c", loss1c_ray == loss1c, "\n")
assert loss1c_ray == loss1c
# </reproduce loss1c multiple loss test on ray>
