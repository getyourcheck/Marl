# %%
from marl.config import Config
trainer_config = Config(
    dict(
        model_path = "facebook/opt-1.3b",
        trainer_type = "huggingface",
        train_kwargs = dict(
            micro_bsz = 1,
            lr = 1e-6,
            total_steps = 1e9,
            lr_decay_rate = 1,
            loss_type = "per_token",
        ),
        parallel = dict(
            data = dict(size = 1),
            tensor = dict(size = 1, mode = "1d"),
            pipeline = dict(size = 1, interleaved_overlap = False),
        ),
    ),
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(trainer_config.get("model_path"))

input_text_list = ["A list of colors: red, blue", "Portugal is"]
inputs = tokenizer(input_text_list, return_tensors="pt", padding=True).to("cuda")
# inputs = { 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0]], device='cuda:0'),
#   'input_ids': tensor([[ 2, 250, 889, 9, 8089, 35, 1275, 6, 2440], [ 2, 22117, 39029, 16, 1, 1, 1, 1, 1]], device='cuda:0') }

# %%
from marl.utils import set_seed
from marl.model_backend.hf_model_trainer import HfModelTrainer
from torch.nn import CrossEntropyLoss
mr = HfModelTrainer(model_config=trainer_config)

input_ids = inputs.input_ids  # shape: [2, 9]
set_seed(1234)
loss1a = mr.compute_loss(input_ids, input_ids)
print('[main]', loss1a)

set_seed(1234)
loss1b = mr.compute_loss(input_ids, input_ids, criterion=CrossEntropyLoss)
print('[main]', loss1b)

print('[main]', "loss1a == loss1b: ", loss1a == loss1b)
assert loss1a == loss1b

set_seed(1234)
input_ids_0 = input_ids[0].unsqueeze(dim=0)  # shape: [1, 9]
input_ids_1 = input_ids[1].unsqueeze(dim=0)  # shape: [1, 9]
loss1c = mr.compute_loss(
    input_ids=[input_ids_0, input_ids_1],
    labels=[input_ids_0, input_ids_1],
    attention_mask=[None, None],
    criterion=[None, None],
    loss_weights=[1.0, 1.0],
)
print('[main]', loss1c)

set_seed(1234)
loss1d1 = mr.compute_loss(input_ids_0, input_ids_0)
loss1d2 = mr.compute_loss(input_ids_1, input_ids_1)
loss1d = (loss1d1 * 1.0 + loss1d2 * 1.0) / 2
print('[main]', loss1d)

print('[main]', "loss1c == loss1d: ", loss1c == loss1d)
assert loss1c == loss1d

# %%
import ray
from marl.model_backend.hf_model_trainer import HfModelTrainerRayActor

set_seed(1234)
ray.init(ignore_reinit_error=True)
mra = HfModelTrainerRayActor.remote(trainer_config)

input_ids = inputs.input_ids  # shape: [2, 9]

loss1a_r = ray.get(mra.compute_loss.remote(input_ids, input_ids))
print('[main]', loss1a_r)

loss1c_r = ray.get(mra.compute_loss.remote(
    input_ids=[input_ids_0, input_ids_1],
    labels=[input_ids_0, input_ids_1],
    attention_mask=[None, None],
    criterion=[None, None],
    loss_weights=[1.0, 1.0],
))
print('[main]', loss1c_r)

# Ray causes some randomness that makes following statements False
print('[main]', f"loss1a_r {loss1a_r} == loss1a {loss1a}: ", loss1a_r == loss1a)
print('[main]', f"loss1c_r {loss1c_r} == loss1c {loss1c}: ", loss1c_r == loss1c)
