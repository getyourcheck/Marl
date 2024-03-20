import torch
from marl.config import Config
from marl.config_consts import ENGINE_HUGGINGFACE
from marl.model_backend.hf_model_runner import HfModelRunner
from marl.utils import set_seed
from marl.loss.actor_loss import ActorLoss
from marl.loss.pretrain_loss import PretrainLoss
from marl.model_backend.generate_utils import (
    partition_by_micro_batch_size, 
    partition_label_by_micro_batch_size,
    partition_list_by_micro_batch_size,
    merge_loss_list
)

def test_partition_by_micro_batch_size():
    # attention_mask None
    input_ids = torch.tensor([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
    attention_mask = None

    micro_batches = partition_by_micro_batch_size(input_ids,-1,attention_mask)
    assert len(micro_batches) == 1
    assert torch.equal(micro_batches[0]["input_ids"],input_ids)
    assert micro_batches[0]["attention_mask"] == None

    micro_batches = partition_by_micro_batch_size(input_ids,1,attention_mask)
    assert len(micro_batches) == 5
    for i in range(5):
        assert torch.equal(micro_batches[i]["input_ids"], torch.tensor([[i for _ in  range(3)]]))
        assert micro_batches[i]["attention_mask"] == None
    
    micro_batches = partition_by_micro_batch_size(input_ids,2,attention_mask)
    assert len(micro_batches) == 3
    for i in range(3):
        assert micro_batches[i]["attention_mask"] == None
    assert torch.equal(micro_batches[0]["input_ids"],torch.tensor([[0,0,0],[1,1,1]]))
    assert torch.equal(micro_batches[1]["input_ids"],torch.tensor([[2,2,2],[3,3,3]]))
    assert torch.equal(micro_batches[2]["input_ids"],torch.tensor([[4,4,4]]))

    micro_batches = partition_by_micro_batch_size(input_ids,3,attention_mask)
    assert len(micro_batches) == 2
    for i in range(2):
        assert micro_batches[i]["attention_mask"] == None
    assert torch.equal(micro_batches[0]["input_ids"],torch.tensor([[0,0,0],[1,1,1],[2,2,2]]))
    assert torch.equal(micro_batches[1]["input_ids"],torch.tensor([[3,3,3],[4,4,4]]))

    micro_batches = partition_by_micro_batch_size(input_ids,5,attention_mask)
    assert len(micro_batches) == 1
    assert torch.equal(micro_batches[0]["input_ids"],input_ids)
    assert micro_batches[0]["attention_mask"] == None

    micro_batches = partition_by_micro_batch_size(input_ids,6,attention_mask)
    assert len(micro_batches) == 1
    assert torch.equal(micro_batches[0]["input_ids"],input_ids)
    assert micro_batches[0]["attention_mask"] == None

    # attention_mask not None
    attention_mask = torch.tensor([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4]])

    micro_batches = partition_by_micro_batch_size(input_ids,-1,attention_mask)
    assert len(micro_batches) == 1
    assert torch.equal(micro_batches[0]["input_ids"],input_ids)
    assert torch.equal(micro_batches[0]["attention_mask"],attention_mask)

    micro_batches = partition_by_micro_batch_size(input_ids,1,attention_mask)
    assert len(micro_batches) == 5
    for i in range(5):
        assert torch.equal(micro_batches[i]["input_ids"], torch.tensor([[i for _ in  range(3)]]))
        assert torch.equal(micro_batches[i]["attention_mask"], torch.tensor([[i for _ in  range(3)]]))
    
    micro_batches = partition_by_micro_batch_size(input_ids,2,attention_mask)
    assert len(micro_batches) == 3
    assert torch.equal(micro_batches[0]["input_ids"],torch.tensor([[0,0,0],[1,1,1]]))
    assert torch.equal(micro_batches[0]["attention_mask"],torch.tensor([[0,0,0],[1,1,1]]))
    assert torch.equal(micro_batches[1]["input_ids"],torch.tensor([[2,2,2],[3,3,3]]))
    assert torch.equal(micro_batches[1]["attention_mask"],torch.tensor([[2,2,2],[3,3,3]]))
    assert torch.equal(micro_batches[2]["input_ids"],torch.tensor([[4,4,4]]))
    assert torch.equal(micro_batches[2]["attention_mask"],torch.tensor([[4,4,4]]))

    micro_batches = partition_by_micro_batch_size(input_ids,3,attention_mask)
    assert len(micro_batches) == 2
    assert torch.equal(micro_batches[0]["input_ids"],torch.tensor([[0,0,0],[1,1,1],[2,2,2]]))
    assert torch.equal(micro_batches[0]["attention_mask"],torch.tensor([[0,0,0],[1,1,1],[2,2,2]]))
    assert torch.equal(micro_batches[1]["input_ids"],torch.tensor([[3,3,3],[4,4,4]]))
    assert torch.equal(micro_batches[1]["attention_mask"],torch.tensor([[3,3,3],[4,4,4]]))

    micro_batches = partition_by_micro_batch_size(input_ids,5,attention_mask)
    assert len(micro_batches) == 1
    assert torch.equal(micro_batches[0]["input_ids"],input_ids)
    assert torch.equal(micro_batches[0]["attention_mask"],attention_mask)

    micro_batches = partition_by_micro_batch_size(input_ids,6,attention_mask)
    assert len(micro_batches) == 1
    assert torch.equal(micro_batches[0]["input_ids"],input_ids)
    assert torch.equal(micro_batches[0]["attention_mask"],attention_mask)

def test_partition_label_by_micro_batch_size():
    labels = torch.Tensor([0,1,2,3,4])
    labels_split = partition_label_by_micro_batch_size(labels,2)
    assert len(labels_split) == 3
    assert torch.equal(labels_split[0],torch.Tensor([0,1]))
    assert torch.equal(labels_split[1],torch.Tensor([2,3]))
    assert torch.equal(labels_split[2],torch.Tensor([4]))
    labels = [torch.Tensor([i]) for i in range(5)]
    labels_split = partition_label_by_micro_batch_size(labels,2)
    assert len(labels_split) == 3
    assert labels_split[0][0] == torch.Tensor([0])
    assert labels_split[0][1] == torch.Tensor([1])
    assert labels_split[1][0] == torch.Tensor([2])
    assert labels_split[1][1] == torch.Tensor([3])
    assert labels_split[2][0] == torch.Tensor([4])
    labels = {
        "key1":torch.Tensor([0,1,2,3,4]),
        "key2":torch.Tensor([0,1,2,3,4]),
        "loss_factor":torch.Tensor([0]),
    }
    labels_split = partition_label_by_micro_batch_size(labels,2,num_splits=3)
    assert len(labels_split) == 3
    assert torch.equal(labels_split[0]['key1'],torch.Tensor([0,1]))
    assert torch.equal(labels_split[1]['key1'],torch.Tensor([2,3]))
    assert torch.equal(labels_split[2]['key1'],torch.Tensor([4]))
    assert torch.equal(labels_split[0]['key2'],torch.Tensor([0,1]))
    assert torch.equal(labels_split[1]['key2'],torch.Tensor([2,3]))
    assert torch.equal(labels_split[2]['key2'],torch.Tensor([4]))
    assert torch.equal(labels_split[0]['loss_factor'],torch.Tensor([0]))
    assert torch.equal(labels_split[1]['loss_factor'],torch.Tensor([0]))
    assert torch.equal(labels_split[2]['loss_factor'],torch.Tensor([0]))

def test_partition_list_by_micro_batch_size():
    input_ids=[torch.tensor([0,1,2,3,4]),torch.tensor([5,6,7,8,9])]
    micro_batch_size=[3,3]
    labels=[
        {
            "key1":torch.Tensor([0,1,2,3,4]),
            "loss_factor":torch.Tensor([0]),
        },
        torch.tensor([0,1,2,3,4])
    ]
    attention_mask=[torch.tensor([0,1,2,3,4]),torch.tensor([5,6,7,8,9])]
    loss_weights=[1,2]
    micro_batches = partition_list_by_micro_batch_size(input_ids,micro_batch_size,labels,attention_mask,loss_weights)
    assert len(micro_batches) == 2
    assert torch.equal(micro_batches[0][0]["input_ids"], torch.tensor([0,1,2]))
    assert torch.equal(micro_batches[1][0]["input_ids"], torch.tensor([3,4]))
    assert torch.equal(micro_batches[0][1]["input_ids"], torch.tensor([5,6,7]))
    assert torch.equal(micro_batches[1][1]["input_ids"], torch.tensor([8,9]))
    assert torch.equal(micro_batches[0][0]["labels"]["key1"], torch.tensor([0,1,2]))
    assert torch.equal(micro_batches[1][0]["labels"]["key1"], torch.tensor([3,4]))
    assert torch.equal(micro_batches[0][0]["labels"]["loss_factor"], torch.Tensor([0]))
    assert torch.equal(micro_batches[1][0]["labels"]["loss_factor"], torch.Tensor([0]))
    assert torch.equal(micro_batches[0][1]["labels"], torch.Tensor([0,1,2]))
    assert torch.equal(micro_batches[1][1]["labels"], torch.Tensor([3,4]))
    assert torch.equal(micro_batches[0][0]["attention_mask"], torch.tensor([0,1,2]))
    assert torch.equal(micro_batches[1][0]["attention_mask"], torch.tensor([3,4]))
    assert torch.equal(micro_batches[0][1]["attention_mask"], torch.tensor([5,6,7]))
    assert torch.equal(micro_batches[1][1]["attention_mask"], torch.tensor([8,9]))
    assert micro_batches[0][0]["loss_weights"] == 1
    assert micro_batches[1][0]["loss_weights"] == 1
    assert micro_batches[0][1]["loss_weights"] == 2
    assert micro_batches[1][1]["loss_weights"] == 2
    input_ids=[torch.tensor([0,1,2,3,4]),torch.tensor([5,6,7,8,9,10,11,12,13,14])]
    micro_batch_size=[3,6]
    labels=[
        {
            "key1":torch.Tensor([0,1,2,3,4]),
            "loss_factor":torch.Tensor([0]),
        },
        torch.tensor([5,6,7,8,9,10,11,12,13,14])
    ]
    attention_mask=[torch.tensor([0,1,2,3,4]),torch.tensor([5,6,7,8,9,10,11,12,13,14])]
    loss_weights=[1,2]
    micro_batches = partition_list_by_micro_batch_size(input_ids,micro_batch_size,labels,attention_mask,loss_weights)
    assert len(micro_batches) == 2
    assert torch.equal(micro_batches[0][0]["input_ids"], torch.tensor([0,1,2]))
    assert torch.equal(micro_batches[1][0]["input_ids"], torch.tensor([3,4]))
    assert torch.equal(micro_batches[0][1]["input_ids"], torch.tensor([5,6,7,8,9,10]))
    assert torch.equal(micro_batches[1][1]["input_ids"], torch.tensor([11,12,13,14]))
    assert torch.equal(micro_batches[0][0]["labels"]["key1"], torch.tensor([0,1,2]))
    assert torch.equal(micro_batches[1][0]["labels"]["key1"], torch.tensor([3,4]))
    assert torch.equal(micro_batches[0][0]["labels"]["loss_factor"], torch.Tensor([0]))
    assert torch.equal(micro_batches[1][0]["labels"]["loss_factor"], torch.Tensor([0]))
    assert torch.equal(micro_batches[0][1]["labels"], torch.Tensor([5,6,7,8,9,10]))
    assert torch.equal(micro_batches[1][1]["labels"], torch.Tensor([11,12,13,14]))
    assert torch.equal(micro_batches[0][0]["attention_mask"], torch.tensor([0,1,2]))
    assert torch.equal(micro_batches[1][0]["attention_mask"], torch.tensor([3,4]))
    assert torch.equal(micro_batches[0][1]["attention_mask"], torch.tensor([5,6,7,8,9,10]))
    assert torch.equal(micro_batches[1][1]["attention_mask"], torch.tensor([11,12,13,14]))
    assert micro_batches[0][0]["loss_weights"] == 1
    assert micro_batches[1][0]["loss_weights"] == 1
    assert micro_batches[0][1]["loss_weights"] == 2
    assert micro_batches[1][1]["loss_weights"] == 2



def test_merge_loss_list():
    loss_list = [
        [torch.tensor([100]),torch.tensor([300])],
        [torch.tensor([200]),torch.tensor([400])]
    ]
    merged = merge_loss_list(loss_list)
    assert len(merged) == 2
    assert merged[0].item() == 150
    assert merged[1].item() == 350

def test_partition_generate_and_infer():
    step = 10
    trainer_config = Config(
        dict(
            model_path="internlm/internlm2-chat-1_8b-sft",
            torch_dtype=torch.float16,
            model_type="actor",
            trainer_type=ENGINE_HUGGINGFACE,
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
    runner = HfModelRunner(model_config=trainer_config)
    runner.initialize()
    input_strs = ["你好", "请提供三个管理时间的建议。"]
    generate_kwargs = {
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
    set_seed(1234)
    policy_output_1 = runner.generate(
        [input_strs[0]],
        step=step,
        output_str=True,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        debug=True,
        generate_kwargs=generate_kwargs,
    )
    set_seed(1234)
    policy_output_2 = runner.generate(
        [input_strs[1]],
        step=step,
        output_str=True,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        debug=True,
        generate_kwargs=generate_kwargs,
    )
    set_seed(1234)
    policy_output_3 = runner.generate(
        input_strs,
        micro_batch_size = 1,
        step=step,
        output_str=True,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        debug=True,
        generate_kwargs=generate_kwargs,
    )
    # question_mask
    left_padding_len = policy_output_3['question_mask'][0].shape[0] - policy_output_1["question_mask"].shape[1]
    assert torch.equal(policy_output_1["question_mask"][0],policy_output_3['question_mask'][0][left_padding_len:])
    assert torch.equal(policy_output_2["question_mask"][0],policy_output_3['question_mask'][1])
    # answer_mask
    assert torch.equal(policy_output_1["answer_mask"][0],policy_output_3['answer_mask'][0][left_padding_len:])
    assert torch.equal(policy_output_2["answer_mask"][0],policy_output_3['answer_mask'][1])
    # attention_mask
    assert torch.equal(policy_output_1["attention_mask"][0],policy_output_3['attention_mask'][0][left_padding_len:])
    assert torch.equal(policy_output_2["attention_mask"][0],policy_output_3['attention_mask'][1])
    # output_ids
    assert torch.equal(policy_output_1["output_ids"][0],policy_output_3['output_ids'][0][left_padding_len:])
    assert torch.equal(policy_output_2["output_ids"][0],policy_output_3['output_ids'][1])
    # output_str
    assert policy_output_1["output_str"][0] == policy_output_3['output_str'][0]
    assert policy_output_2["output_str"][0] == policy_output_3['output_str'][1]
    # output_ans_str
    assert policy_output_1["output_ans_str"][0] == policy_output_3['output_ans_str'][0]
    assert policy_output_2["output_ans_str"][0] == policy_output_3['output_ans_str'][1]

    output_ids = policy_output_3['output_ids']
    attention_mask = policy_output_3['attention_mask']

    set_seed(1234)
    policy_output_4 = runner.infer(output_ids[0].unsqueeze(0), attention_mask=attention_mask[0].unsqueeze(0), output_logprobs=True, debug=True)
    set_seed(1234)
    policy_output_5 = runner.infer(output_ids[1].unsqueeze(0), attention_mask=attention_mask[1].unsqueeze(0), output_logprobs=True, debug=True)
    set_seed(1234)
    policy_output_6 = runner.infer(output_ids, attention_mask=attention_mask, output_logprobs=True, debug=True, micro_batch_size = 1)
    assert torch.equal(policy_output_4["logprobs"][0],policy_output_6['logprobs'][0])
    assert torch.equal(policy_output_5["logprobs"][0],policy_output_6['logprobs'][1])

def test_train():
    def weighted_loss(train_loss, pretrain_loss, loss_weights):
        loss_weights = [x / float(len(loss_weights)) for x in [1,2]]  # to 1
        return train_loss * loss_weights[0] + pretrain_loss * loss_weights[1]
    trainer_config = Config(
        dict(
            model_path="internlm/internlm2-chat-1_8b-sft",
            torch_dtype=torch.float16,
            model_type="actor",
            trainer_type=ENGINE_HUGGINGFACE,
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
    mr = HfModelRunner(model_config=trainer_config)
    mr.initialize()
    input_ids = torch.tensor([[7558,  1244,  4795,  9551,   446,   395, 48929,   262,  1238,  1184,
            2460,   579,   940,  6022,   454, 31449,   328,   607,   784,   629,
            1896,   697,   725,  2320,  1263,   884,  5426,   333,   352, 23845,
            352, 27232,   489],
            [6666,  1244,  4795,  9551,   446,   395, 48929,   262,  1238,  1184,
            2460,   579,   940,  6022,   454, 31449,   328,   607,   784,   629,
            1896,   697,   725,  2320,  1263,   884,  5426,   333,   352, 23845,
            352, 27232,   489]]).cuda()
    log_probs = torch.tensor([[0, -5.9688,  -7.6875,  -4.5000,  -0.0232,  -2.4375,  -4.7500,  -4.2812,
        -1.6406,  -0.1846,  -0.0232,  -5.3438,  -4.7500,  -6.2188,  -1.4219,
        -2.5625,  -1.9062,  -4.3125,  -4.7500,  -0.0605,  -0.7461,  -1.2734,
        -2.2031,  -2.3438,  -2.0312,  -2.4062,  -2.6562,  -7.9375,  -6.3125,
        -13.2500,  -2.0312,  -5.0625,  -1.2031],
        [0, -5.9688,  -7.6875,  -4.5000,  -0.0232,  -2.4375,  -4.7500,  -4.2812,
        -1.6406,  -0.1846,  -0.0232,  -5.3438,  -4.7500,  -6.2188,  -1.4219,
        -2.5625,  -1.9062,  -4.3125,  -4.7500,  -0.0605,  -0.7461,  -1.2734,
        -2.2031,  -2.3438,  -2.0312,  -2.4062,  -2.6562,  -7.9375,  -6.3125,
        -13.2500,  -2.0312,  -5.0625,  -1.2031]],dtype=torch.bfloat16).cuda()
    advantages = torch.tensor([[0, 224.,  752.,  366.,  492.,  596.,  340.,  416.,  652.,  832., 1012.,
        624.,  -21., -152.,  880.,  196.,  856.,  628.,  860.,  704., 1112.,
        410.,  524.,  912., 1056.,  480., 1040.,  584.,  362., -204.,  124.,
        -138.,  376.],
        [0, 224.,  752.,  366.,  492.,  596.,  340.,  416.,  652.,  832., 1012.,
        624.,  -21., -152.,  880.,  196.,  856.,  628.,  860.,  704., 1112.,
        410.,  524.,  912., 1056.,  480., 1040.,  584.,  362., -204.,  124.,
        -138.,  376.]],dtype=torch.bfloat16).cuda()
    response_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1 ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1 ]]).cuda()
    valid_tokens = response_mask.sum().cuda()
    loss_factor = 1 / valid_tokens.item()
    labels_0 = dict(
        input_ids=input_ids[0].unsqueeze(0),
        old_logprobs=log_probs[0].unsqueeze(0),
        advantages=advantages[0].unsqueeze(0),
        mask=response_mask[0].unsqueeze(0),
        loss_factor=torch.tensor(loss_factor),
    )
    labels_1 = dict(
        input_ids=input_ids[1].unsqueeze(0),
        old_logprobs=log_probs[0].unsqueeze(0),
        advantages=advantages[0].unsqueeze(0),
        mask=response_mask[0].unsqueeze(0),
        loss_factor=torch.tensor(loss_factor),
    )
    labels = dict(
        input_ids=input_ids,
        old_logprobs=log_probs,
        advantages=advantages,
        mask=response_mask,
        loss_factor=torch.tensor(loss_factor),
    )
    set_seed(1234)
    loss_1 = mr.train(input_ids[0].unsqueeze(0), labels_0, criterion=ActorLoss(), step_interval=999).item()
    set_seed(1234)
    loss_2 = mr.train(input_ids[1].unsqueeze(0), labels_1, criterion=ActorLoss(), step_interval=999).item()
    set_seed(1234)
    loss_3 = mr.train(input_ids, labels, criterion=ActorLoss(), step_interval=999, micro_batch_size=1).item()
    assert loss_3 == (loss_1 + loss_2) / 2

    pretrain_input_ids = torch.tensor([[ 611, 18963,  1893, 15065,  3126,   491, 10850,   454, 56025, 19603,
           756,   918,  6690,   435, 28260,  5992,  1556,   668,   517,   937,
          2308,   281, 18590,   518,   451, 21239,  8726,   454,  8917,   313,
          7175, 34209,   569],
          [ 888, 18963,  1893, 15065,  3126,   491, 10850,   454, 56025, 19603,
           756,   918,  6690,   435, 28260,  5992,  1556,   668,   517,   937,
          2308,   281, 18590,   518,   451, 21239,  8726,   454,  8917,   313,
          7175, 34209,   569]]).cuda()
    pretrain_labels = torch.tensor([[ 18963,  1893, 15065,  3126,   491, 10850,   454, 56025, 19603,
           756,   918,  6690,   435, 28260,  5992,  1556,   668,   517,   937,
          2308,   281, 18590,   518,   451, 21239,  8726,   454,  8917,   313,
          7175, 34209,   569,   2792],
          [ 18888,  1893, 15065,  3126,   491, 10850,   454, 56025, 19603,
           756,   918,  6690,   435, 28260,  5992,  1556,   668,   517,   937,
          2308,   281, 18590,   518,   451, 21239,  8726,   454,  8917,   313,
          7175, 34209,   569,   2792]]).cuda()
    set_seed(1234)
    train_loss_4, pre_train_loss_4 = mr.train(
        [input_ids[0].unsqueeze(0),pretrain_input_ids[0].unsqueeze(0)], 
        [labels_0,pretrain_labels[0].unsqueeze(0)], 
        criterion=[ActorLoss(),PretrainLoss()], 
        loss_weights=[1,2],
        attention_mask = [None,None],
        step_interval=999)
    loss_4 = weighted_loss(train_loss_4.item(),pre_train_loss_4.item(),[1,2])
    set_seed(1234)
    train_loss_5, pre_train_loss_5 = mr.train(
        [input_ids[1].unsqueeze(0),pretrain_input_ids[1].unsqueeze(0)], 
        [labels_1,pretrain_labels[1].unsqueeze(0)], 
        criterion=[ActorLoss(),PretrainLoss()], 
        loss_weights=[1,2],
        attention_mask = [None,None],
        step_interval=999)
    loss_5 = weighted_loss(train_loss_5.item(),pre_train_loss_5.item(),[1,2])
    set_seed(1234)
    train_loss_6,pre_train_loss_6 = mr.train(
        [input_ids,pretrain_input_ids], 
        [labels,pretrain_labels], 
        criterion=[ActorLoss(),PretrainLoss()], 
        step_interval=999,
        loss_weights=[1,2],
        attention_mask = [None,None],
        micro_batch_size=[1,1],
        debug=True)
    loss_6 = weighted_loss(train_loss_6.item(),pre_train_loss_6.item(),[1,2])
    print(loss_4)
    print(loss_5)
    print(loss_6)
    print((loss_5 + loss_4) / 2)

    assert round(loss_6,5) == round((loss_5 + loss_4) / 2 ,5)