# %%
from marl.config.config import Config
from marl.model_backend.vllm_model_runner import VllmGenerator
from marl.model_backend.hf_model_runner import HfModelRunner
from marl.timer import Timer
from marl.policy_output import PolicyOutput
import torch 

config = Config.from_file("projects/ppo/internlm2/1B/actor_1gpu.py")
config["actor"]["generator_config"]["generate_kwargs"]["top_k"] = 1
config["actor"]["generator_config"]["generate_kwargs"]["early_stopping"] = False

actor_config = config['actor']
actor_model = VllmGenerator(actor_config)
actor_model.initialize()

input_ids = torch.tensor([[    0,     0,     0,   525, 11353,   364],
                          [    0,     0,     0,   525, 11353,   364]], dtype=int)


policy_output: PolicyOutput = actor_model.generate(
    input_ids,
    step=10,
    output_logits=False,
    output_str=True,
    generate_kwargs=config["actor"]["generator_config"]["generate_kwargs"],
    attention_mask=None,
)
print(policy_output.output_ids)
print(policy_output.output_str)
print(policy_output.answer_mask)
print(policy_output.question_mask)

# %%
max_answer_len = 4096
prompt = "请输出 10000 遍 hello"
step = max_answer_len - 19  # 19 is number of tokens of encoded prompt

micro_batch_size = 1
while True:
    input_strs = [prompt for _ in range(micro_batch_size)]
    # input_strs = [[{"role": "user", "content": prompt}] for _ in range(micro_batch_size)] # TODO

    with Timer(f"vLLMGenerator generate with micro_batch_size={micro_batch_size}") as t:
        policy_output: PolicyOutput = actor_model.generate(
            input_strs,
            step=step,
            output_logits=False,
            output_str=True,
            generate_kwargs=config["actor"]["generator_config"]["generate_kwargs"],
            attention_mask=None,
        )

    if micro_batch_size == 1:
        print(policy_output.output_str[0])

    print(f"micro_batch_size:[{micro_batch_size}] generate takes {t.duration} seconds")
    micro_batch_size *= 2

exit()
# %%
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="internlm/internlm2-chat-1_8b-sft", trust_remote_code=True)

output_ids = llm.generate(prompts, sampling_params=sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
