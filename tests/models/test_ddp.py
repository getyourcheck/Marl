from marl.coordinator import Coordinator
from marl.config import Config
from marl.tokenizer.tokenizer_utils import get_tokenizer
from marl.loss.actor_loss import ActorLoss
import torch
import ray
import pytest
from marl.policy_output import concat_policy_outputs

# @pytest.mark.skip()
def test_generate_ddp():
    cluster_address = "auto"
    input_strs = ["你好", "床前明月光", "你好", "床前明月光"]
    input_strs = [[{"role": "user", "content": input_str}] for input_str in input_strs]

    config_2dp = Config.from_file("projects/ppo/internlm2/1B/actor_2gpu.py")
    coordinator_2dp = Coordinator(cluster_address, config_2dp)
    model_dict_2dp = coordinator_2dp.create_models()
    actor_model_2dp = model_dict_2dp["actor"]

    actor_model_2dp.trainer.set_seed(1234)
    output_2dp = actor_model_2dp.generate(
        input_strs,
        step=50,
        output_logits=False,
        output_str=True,
        generate_kwargs=config_2dp["actor"]["generator_config"]["generate_kwargs"],
        attention_mask=None
    )
    output_ids_2dp = output_2dp.output_ids
    coordinator_2dp.clean_up()

    config_1dp = Config.from_file("projects/ppo/internlm2/1B/actor_1gpu.py")
    coordinator_1dp = Coordinator(cluster_address, config_1dp)
    model_dict_1dp = coordinator_1dp.create_models()
    actor_model_1dp = model_dict_1dp["actor"]

    actor_model_1dp.trainer.set_seed(1234)
    output_1dp_0 = actor_model_1dp.generate(
        input_strs[0:2],
        step=50,
        output_logits=False,
        output_str=True,
        generate_kwargs=config_1dp["actor"]["generator_config"]["generate_kwargs"],
        attention_mask=None
    )

    actor_model_1dp.trainer.set_seed(1234)
    output_1dp_1 = actor_model_1dp.generate(
        input_strs[2:4],
        step=50,
        output_logits=False,
        output_str=True,
        generate_kwargs=config_1dp["actor"]["generator_config"]["generate_kwargs"],
        attention_mask=None
    )
    coordinator_1dp.clean_up()
    model_path = "internlm/internlm2-chat-1_8b-sft"
    tokenizer = get_tokenizer(model_path, trust_remote_code=True)
    padding_token_map = {"output_ids":tokenizer.unk_token_id}
    output_1dp = concat_policy_outputs([output_1dp_0, output_1dp_1], padding_token_map=padding_token_map)
    output_ids_1dp = output_1dp.output_ids

    assert torch.equal(output_ids_2dp, output_ids_1dp)
    
# @pytest.mark.skip()
def test_actor_infer_ddp():
    tokenizer = get_tokenizer("internlm/internlm2-chat-1_8b-sft", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token
    input_strs=[
        "两个黄鹂鸣翠柳",
        "两个黄鹂鸣翠柳，一行白鹭上青天",
        "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪",
        "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    ]
    test_data = tokenizer(input_strs, return_tensors="pt", padding=True)
    input_ids = test_data.input_ids
    attention_mask = test_data.attention_mask
    cluster_address = "auto"
    config_2dp = Config.from_file("projects/ppo/internlm2/1B/actor_2gpu.py")
    coordinator_2dp = Coordinator(cluster_address, config_2dp)
    model_dict_2dp = coordinator_2dp.create_models()
    actor_model_2dp = model_dict_2dp["actor"]
    output_2dp = actor_model_2dp.infer(input_ids, attention_mask=attention_mask, output_logprobs=True)
    logprobs_2dp = output_2dp["logprobs"]
    coordinator_2dp.clean_up()
    
    config_1dp = Config.from_file("projects/ppo/internlm2/1B/actor_1gpu.py")
    coordinator_1dp = Coordinator(cluster_address, config_1dp)
    model_dict_1dp = coordinator_1dp.create_models()
    actor_model_1dp = model_dict_1dp["actor"]
    output_1dp_0 = actor_model_1dp.infer(input_ids[0:2,:], attention_mask=attention_mask[0:2,:], output_logprobs=True)
    logprobs_1dp_0 = output_1dp_0["logprobs"]
    output_1dp_1 = actor_model_1dp.infer(input_ids[2:4,:], attention_mask=attention_mask[2:4,:], output_logprobs=True)
    logprobs_1dp_1 = output_1dp_1["logprobs"]
    output_1dp = torch.cat([logprobs_1dp_0,logprobs_1dp_1],dim=0)
    coordinator_1dp.clean_up()

    assert torch.equal(logprobs_2dp, output_1dp)

# @pytest.mark.skip()
def test_train_ddp():
    cluster_address = "auto"
    config_2dp = Config.from_file("projects/ppo/internlm2/1B/actor_2gpu.py")
    coordinator_2dp = Coordinator(cluster_address, config_2dp)
    model_dict_2dp = coordinator_2dp.create_models()
    actor_model_2dp = model_dict_2dp["actor"]

    input_ids = torch.tensor([[1111,  1244,  4795,  9551,   446,   395, 48929,   262,  1238,  1184,
            2460,   579,   940,  6022,   454, 31449,   328,   607,   784,   629,
            1896,   697,   725,  2320,  1263,   884,  5426,   333,   352, 23845,
            352, 27232,   489],
            [2222,  1244,  4795,  9551,   446,   395, 48929,   262,  1238,  1184,
            2460,   579,   940,  6022,   454, 31449,   328,   607,   784,   629,
            1896,   697,   725,  2320,  1263,   884,  5426,   333,   352, 23845,
            352, 27232,   489],
            [3333,  1244,  4795,  9551,   446,   395, 48929,   262,  1238,  1184,
            2460,   579,   940,  6022,   454, 31449,   328,   607,   784,   629,
            1896,   697,   725,  2320,  1263,   884,  5426,   333,   352, 23845,
            352, 27232,   489],
            [4444,  1244,  4795,  9551,   446,   395, 48929,   262,  1238,  1184,
            2460,   579,   940,  6022,   454, 31449,   328,   607,   784,   629,
            1896,   697,   725,  2320,  1263,   884,  5426,   333,   352, 23845,
            352, 27232,   489]]).cuda()
    log_probs = torch.tensor([[0, -1.1111,  -7.6875,  -4.5000,  -0.0232,  -2.4375,  -4.7500,  -4.2812,
        -1.6406,  -0.1846,  -0.0232,  -5.3438,  -4.7500,  -6.2188,  -1.4219,
        -2.5625,  -1.9062,  -4.3125,  -4.7500,  -0.0605,  -0.7461,  -1.2734,
        -2.2031,  -2.3438,  -2.0312,  -2.4062,  -2.6562,  -7.9375,  -6.3125,
        -13.2500,  -2.0312,  -5.0625,  -1.2031],
        [0, -2.2222,  -7.6875,  -4.5000,  -0.0232,  -2.4375,  -4.7500,  -4.2812,
        -1.6406,  -0.1846,  -0.0232,  -5.3438,  -4.7500,  -6.2188,  -1.4219,
        -2.5625,  -1.9062,  -4.3125,  -4.7500,  -0.0605,  -0.7461,  -1.2734,
        -2.2031,  -2.3438,  -2.0312,  -2.4062,  -2.6562,  -7.9375,  -6.3125,
        -13.2500,  -2.0312,  -5.0625,  -1.2031],
        [0, -3.3333,  -7.6875,  -4.5000,  -0.0232,  -2.4375,  -4.7500,  -4.2812,
        -1.6406,  -0.1846,  -0.0232,  -5.3438,  -4.7500,  -6.2188,  -1.4219,
        -2.5625,  -1.9062,  -4.3125,  -4.7500,  -0.0605,  -0.7461,  -1.2734,
        -2.2031,  -2.3438,  -2.0312,  -2.4062,  -2.6562,  -7.9375,  -6.3125,
        -13.2500,  -2.0312,  -5.0625,  -1.2031],
        [0, -4.4444,  -7.6875,  -4.5000,  -0.0232,  -2.4375,  -4.7500,  -4.2812,
        -1.6406,  -0.1846,  -0.0232,  -5.3438,  -4.7500,  -6.2188,  -1.4219,
        -2.5625,  -1.9062,  -4.3125,  -4.7500,  -0.0605,  -0.7461,  -1.2734,
        -2.2031,  -2.3438,  -2.0312,  -2.4062,  -2.6562,  -7.9375,  -6.3125,
        -13.2500,  -2.0312,  -5.0625,  -1.2031]],dtype=torch.bfloat16).cuda()
    advantages = torch.tensor([[0, 111.,  752.,  366.,  492.,  596.,  340.,  416.,  652.,  832., 1012.,
        624.,  -21., -152.,  880.,  196.,  856.,  628.,  860.,  704., 1112.,
        410.,  524.,  912., 1056.,  480., 1040.,  584.,  362., -204.,  124.,
        -138.,  376.],
        [0, 222.,  752.,  366.,  492.,  596.,  340.,  416.,  652.,  832., 1012.,
        624.,  -21., -152.,  880.,  196.,  856.,  628.,  860.,  704., 1112.,
        410.,  524.,  912., 1056.,  480., 1040.,  584.,  362., -204.,  124.,
        -138.,  376.],
        [0, 333.,  752.,  366.,  492.,  596.,  340.,  416.,  652.,  832., 1012.,
        624.,  -21., -152.,  880.,  196.,  856.,  628.,  860.,  704., 1112.,
        410.,  524.,  912., 1056.,  480., 1040.,  584.,  362., -204.,  124.,
        -138.,  376.],
        [0, 444.,  752.,  366.,  492.,  596.,  340.,  416.,  652.,  832., 1012.,
        624.,  -21., -152.,  880.,  196.,  856.,  628.,  860.,  704., 1112.,
        410.,  524.,  912., 1056.,  480., 1040.,  584.,  362., -204.,  124.,
        -138.,  376.]],dtype=torch.bfloat16).cuda()
    response_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1 ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1 ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1 ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1 ]]).cuda()
    valid_tokens = response_mask.sum().cuda()
    loss_factor = 1 / valid_tokens.item()
    labels = dict(
        input_ids=input_ids,
        old_logprobs=log_probs,
        advantages=advantages,
        mask=response_mask,
        loss_factor=torch.tensor(loss_factor),
    )

    loss_2dp = actor_model_2dp.train(
        input_ids=input_ids,
        labels=labels,     
        attention_mask=None,
        criterion=ActorLoss())
    model_dp0 = ray.get(actor_model_2dp.trainer.ray_actors[0].get_model.remote())
    model_dp1 = ray.get(actor_model_2dp.trainer.ray_actors[1].get_model.remote())
    coordinator_2dp.clean_up()

    model_dp0_params ={}
    for k,v in model_dp0.named_parameters():
        model_dp0_params[k] = v
    model_dp1_params ={}
    for k,v in model_dp1.named_parameters():
        model_dp1_params[k] = v
    
    for k,v in model_dp0_params.items():
        param_dp0 = v
        param_dp1 = model_dp1_params[k]
        assert(torch.equal(param_dp0,param_dp1))
    
    config_1dp = Config.from_file("projects/ppo/internlm2/1B/actor_1gpu.py")
    coordinator_1dp = Coordinator(cluster_address, config_1dp)
    model_dict_1dp = coordinator_1dp.create_models()
    actor_model_1dp = model_dict_1dp["actor"]

    labels_0 = dict(
        input_ids=input_ids[0:2],
        old_logprobs=log_probs[0:2],
        advantages=advantages[0:2],
        mask=response_mask[0:2],
        loss_factor=torch.tensor(loss_factor),
    )

    labels_1 = dict(
        input_ids=input_ids[2:4],
        old_logprobs=log_probs[2:4],
        advantages=advantages[2:4],
        mask=response_mask[2:4],
        loss_factor=torch.tensor(loss_factor),
    )

    loss_1dp_0 = actor_model_1dp.train(
        input_ids=input_ids[0:2],
        labels=labels_0,     
        attention_mask=None,
        criterion=ActorLoss(),
        step_interval=999
    )
    loss_1dp_1 = actor_model_1dp.train(
        input_ids=input_ids[2:4],
        labels=labels_1,     
        attention_mask=None,
        criterion=ActorLoss(),
        step_interval=999
    )
    coordinator_1dp.clean_up()
    assert loss_2dp == (loss_1dp_0 + loss_1dp_1) / 2
