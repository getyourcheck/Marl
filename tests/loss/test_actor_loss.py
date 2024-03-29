import torch
from marl.config import Config
from marl.config_consts import ENGINE_HUGGINGFACE
from marl.model_backend.hf_model_runner import HfModelRunner
from marl.loss.actor_loss import ActorLoss
from marl.loss.pretrain_loss import PretrainLoss

def test_actor_loss():
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
            352, 27232,   489]]).cuda()
    log_probs = torch.tensor([[0, -5.9688,  -7.6875,  -4.5000,  -0.0232,  -2.4375,  -4.7500,  -4.2812,
        -1.6406,  -0.1846,  -0.0232,  -5.3438,  -4.7500,  -6.2188,  -1.4219,
        -2.5625,  -1.9062,  -4.3125,  -4.7500,  -0.0605,  -0.7461,  -1.2734,
        -2.2031,  -2.3438,  -2.0312,  -2.4062,  -2.6562,  -7.9375,  -6.3125,
        -13.2500,  -2.0312,  -5.0625,  -1.2031]],dtype=torch.bfloat16).cuda()
    advantages = torch.tensor([[0, 224.,  752.,  366.,  492.,  596.,  340.,  416.,  652.,  832., 1012.,
        624.,  -21., -152.,  880.,  196.,  856.,  628.,  860.,  704., 1112.,
        410.,  524.,  912., 1056.,  480., 1040.,  584.,  362., -204.,  124.,
        -138.,  376.]],dtype=torch.bfloat16).cuda()
    response_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
    loss = mr.train(input_ids, labels, criterion=ActorLoss(), step_interval=999).item()
    print(loss)
    assert loss == -117.1788101196289
    pretrain_input_ids = torch.tensor([[ 611, 18963,  1893, 15065,  3126,   491, 10850,   454, 56025, 19603,
           756,   918,  6690,   435, 28260,  5992,  1556,   668,   517,   937,
          2308,   281, 18590,   518,   451, 21239,  8726,   454,  8917,   313,
          7175, 34209,   569]]).cuda()
    pretrain_labels = torch.tensor([[ 18963,  1893, 15065,  3126,   491, 10850,   454, 56025, 19603,
           756,   918,  6690,   435, 28260,  5992,  1556,   668,   517,   937,
          2308,   281, 18590,   518,   451, 21239,  8726,   454,  8917,   313,
          7175, 34209,   569,   2792]]).cuda()
    train_loss, pretrain_loss = mr.train(
        input_ids = [input_ids, pretrain_input_ids], 
        labels = [labels, pretrain_labels], 
        criterion=[ActorLoss(), PretrainLoss()], 
        attention_mask = [None,None],
        loss_weights =[1,1],
        step_interval=999
    )
    assert train_loss.item() == -117.1788101196289
    assert pretrain_loss.item() == 12.587218284606934
