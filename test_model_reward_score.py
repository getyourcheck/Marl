from marl.config import Config
from marl.coordinator import Coordinator
from copy import deepcopy
import re

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
            GLOBAL_STEPS = file.readline() ## GLOBAL_STEPS = 0000 #
            step = int(GLOBAL_STEPS[17:21])
            if step > 0:
                data[step - 1] = array
                array = []
            line = file.readline() ########################
            line = file.readline()
            continue
        if line == '******** Prompt ********\n':
            while line != '******** For Reward ********\n':
                line = file.readline()
            # print(line) #******** For Reward ********
            line = file.readline()
            tmp = ""
            while not line.startswith("reward_score"):
                tmp += line
                line = file.readline()
            tmp=tmp[:-1]
            # print(tmp)
            array.append(tmp)
            continue
        line = file.readline()
    # for k,v in data.items():
    #     print(f"[{k}][{len(v)}]")
    file.close()

    def convert_to_json_array(raw_string):
        pattern = r"(\[UNUSED_TOKEN_146\]user|\[UNUSED_TOKEN_146\]assistant)(.+?)(?=(\[UNUSED_TOKEN_146\]user|\[UNUSED_TOKEN_146\]assistant|$))"
        matches = re.findall(pattern, raw_string, re.DOTALL)
        json_array = []
        for match in matches:
            role, content = match[0], match[1].strip()
            role = role.replace('[UNUSED_TOKEN_146]', '')
            json_array.append({"role": role, "content": content})
        return json_array

    new_data={}
    for k,v in data.items():
        new_v = []
        for strs in v:
            # strs = strs.replace('[UNUSED_TOKEN_146]', '')
            strs = strs.replace('[UNUSED_TOKEN_145]', '')
            strs = strs.replace('[UNUSED_TOKEN_130]', '')
            strs=strs[:-1]
            json_str=convert_to_json_array(strs)
            new_v.append(json_str)
        new_data[k]=new_v
    return new_data

cluster_address="auto"
configs_path = "projects/ppo/internlm2/1B/actor_reward_8gpu.py"
config = Config.from_file(configs_path)
coordinator = Coordinator(cluster_address, config["model_configs"])
model_dict = coordinator.create_models()
actor_model = model_dict["actor"]
reward_model = model_dict["reward"]

data = get_data("/cpfs01/shared/public/llm_model/ckpt/test_zhaohui_0424/1.8B-baseline-c2kcl5.log.txt")

prompt=data[0]
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
    micro_batch_size=16,
    generate_kwargs=generate_kwargs
)

input_messages = deepcopy(prompt)
for i in range(len(range(len(trajectories.output_ans_str)))):
    input_messages[i].append({"role": "assistant", "content": trajectories.output_ans_str[i]})

value_output = reward_model.infer(
    input_messages, 
    output_logprobs=False, 
    micro_batch_size=32,
)
reward = value_output.logits.cpu().mean().item()
print(f"===================[reward:{reward}]")