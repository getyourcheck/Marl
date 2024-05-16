import torch
from marl.config import Config
from marl.coordinator import Coordinator
from marl.config import Config
from marl.model_backend.hf_model_runner import HfModelRunner
import numpy as np
import re
from marl.model_backend.models.internlm2_reward import (
    InternLM2ForRewardModel,
)
torch.set_printoptions(threshold=np.inf)

def get_data(path):
    file = open(path, "r")
    line = file.readline()
    data = {}
    array = []
    while line:
        if line == '\n':
            line = file.readline()
            continue
        if line == '#######################\n':
            GLOBAL_STEPS = file.readline()
            step = int(GLOBAL_STEPS[17:21])
            if step > 0:
                data[step - 1] = array
                array = []
            line = file.readline()
            line = file.readline()
            continue
        if line == '******** Prompt ********\n':
            while line != '******** For Reward ********\n':
                line = file.readline()
            line = file.readline()
            tmp = ""
            while not line.startswith("reward_score"):
                tmp += line
                line = file.readline()
            tmp=tmp[:-1]
            array.append(tmp)
            continue
        line = file.readline()
    file.close()

    def convert_to_json_array(raw_string):
        pattern = r"(user|assistant)(.+?)(?=(user|assistant|$))"
        matches = re.findall(pattern, raw_string, re.DOTALL)
        json_array = []
        for match in matches:
            role, content = match[0], match[1].strip()
            json_array.append({"role": role, "content": content})
        return json_array

    new_data={}
    for k,v in data.items():
        new_v = []
        for strs in v:
            strs = strs.replace('[UNUSED_TOKEN_146]', '')
            strs = strs.replace('[UNUSED_TOKEN_145]', '')
            strs = strs.replace('[UNUSED_TOKEN_130]', '')
            strs=strs[:-1]
            json_str=convert_to_json_array(strs)
            new_v.append(json_str)
        new_data[k]=new_v
    return new_data


cluster_address = "ray://127.0.0.1:10001"
config = Config.from_file("projects/ppo/internlm2/1B/actor_vllm_reward_score.py")
coordinator = Coordinator(cluster_address, config)
model_dict = coordinator.create_models()
actor_model = model_dict["actor"]

reward_config = Config(
    dict(
        # model_path="/fs-computility/llm/shared/marl/models/internlm2/1.8B/hf/R-Luyou-1B-8k-D20240130-v1/819_hf_bf16/",
        model_path="/cpfs01/shared/public/public_hdd/lvchengqi/ckpts/reward_model/R-Luyou-1B-8k-D20240130-v1/819_hf_bf16/",
        model_type="reward",
        model_class=InternLM2ForRewardModel,
        torch_dtype=torch.bfloat16,
        train_kwargs=dict(
            micro_bsz=1,
            lr=5e-6,
            total_steps=1e9,
            lr_decay_rate=1,
            loss_type="per_seq",
        ),
        parallel=dict(
            data=dict(size=2, mode="ddp"),
            tensor=dict(size=1, mode="1d"),
            pipeline=dict(size=1, interleaved_overlap=False),
            sequence=False,
        ),
    ),
)
reward_model = HfModelRunner(model_config=reward_config)
reward_model.initialize()

data = get_data("/cpfs01/shared/public/llm_model/ckpt/test_0416/1.8B-baseline-c2kcl5.log.txt")


for step in range(1):

    prompt=data[step]
    for i in range(len(prompt)):
        prompt[i] = prompt[i][:-1]

    generate_kwargs={
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 0.9,
        "min_new_tokens": 1,
        "num_beams": 1,
        "early_stopping": True,
        "eos_token_id": 92542,
        "pad_token_id": 0,
    }

    trajectories = actor_model.generate(
        inputs=prompt,
        step=1024,
        output_str=True, 
        micro_batch_size=64,
        generate_kwargs=generate_kwargs
    )

    value_output = reward_model.infer(
        inputs=trajectories.output_ids,
        attention_mask=trajectories.attention_mask, 
        output_logprobs=False, 
        micro_batch_size=8,
    )
    # value_output = ray.get(ref)
    reward = value_output.logits.cpu().mean()

    print(f"===================[step:{step}][reward:{reward}]")