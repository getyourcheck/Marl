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
from marl.model_backend.hf_model_trainer import HfModelTrainer
mr = HfModelTrainer(model_config=trainer_config)

loss = mr.compute_loss(inputs.input_ids, inputs.input_ids)
print(loss)

loss2 = mr.compute_loss(inputs.input_ids, inputs.input_ids)
print(loss2)

# %%
import ray
from marl.model_backend.hf_model_trainer import HfModelTrainerRayActor

ray.init()
mra = HfModelTrainerRayActor.remote(trainer_config)

infer_config = {"max_new_tokens": 64}
output_inf_r = ray.get(mra.infer.remote(inputs.input_ids, step=1, **infer_config))
print(output_inf_r)

output_gen_r = ray.get(mra.infer.remote(inputs.input_ids, **infer_config))
print(output_gen_r)

# %%
print("output_inf_r == output_inf: ", output_inf_r == output_inf)
print("output_gen_r == output_gen: ", output_gen_r == output_gen)

# %%
