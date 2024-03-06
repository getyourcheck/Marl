# %%
# 0. imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# %%
# 1. load a pretrained model
# MODEL_PATH="internlm/internlm2-7b"
MODEL_PATH="facebook/opt-1.3b"
model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True).cuda()
model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True).cuda()
model_ref = model_ref.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# %%
# 2. initialize trainer
ppo_config = {"batch_size": 2, "mini_batch_size": 2}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer)

# 3. encode a query
query_txt = ["This morning I went to the ", "A list of colors: red, blue"]
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)

# 4. generate model response
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
    "generate_ref_response": True,
}
response_tensor, ref_response = ppo_trainer.generate([item for item in query_tensor], return_prompt=False, **generation_kwargs)
response_txt = tokenizer.decode(response_tensor[0])

# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

# 6. train model with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
