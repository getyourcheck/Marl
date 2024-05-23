
import torch
import pytest
import tempfile
import shutil
from transformers import AutoModelForCausalLM
from marl.config.config import Config
from marl.model_backend.hf_model_runner import HfModelRunner
from marl.coordinator import Coordinator
from marl.model_server.base_model_server import BaseModelServer


@pytest.mark.skip()
def check_model(model,target_keys:dict):
    model_param_map = {}
    for k, _ in model.named_parameters():
        model_param_map[k]=k
    for k in target_keys:
        assert k in model_param_map

@pytest.mark.skip()
def check_state_dict(state_dict:dict,target_keys:dict):
    model_param_map = {}
    for k, _ in state_dict.items():
        model_param_map[k]=k
    for k in target_keys:
        assert k in model_param_map

@pytest.mark.skip()
def get_target_keys():
    target_keys=[]
    target_keys.append("model.tok_embeddings.weight")
    for i in range(24):
        target_keys.append(f"model.layers.{i}.attention.wqkv.weight")
        target_keys.append(f"model.layers.{i}.attention.wo.weight")
        target_keys.append(f"model.layers.{i}.feed_forward.w1.weight")
        target_keys.append(f"model.layers.{i}.feed_forward.w3.weight")
        target_keys.append(f"model.layers.{i}.feed_forward.w2.weight")
        target_keys.append(f"model.layers.{i}.attention_norm.weight")
        target_keys.append(f"model.layers.{i}.ffn_norm.weight")
    target_keys.append("model.norm.weight")
    target_keys.append("output.weight")
    return target_keys

# @pytest.mark.skip()
def test_no_ray():
    target_keys = get_target_keys()
    # no ray
    trainer_config = Config(
        dict(
            model_path="internlm/internlm2-chat-1_8b-sft",
            model_type="actor",
            torch_dtype=torch.bfloat16,
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
                loss_type="per_seq",
            ),
            parallel=dict(
                data=dict(size=1, mode="ddp"),
                tensor=dict(size=1, mode="1d"),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=False,
            ),
        ),
    )
    mr = HfModelRunner(model_config=trainer_config)
    mr.initialize()

    model = mr.get_model()
    check_model(model,target_keys)
    del model
    state_dict = mr.get_state_dict()
    check_state_dict(state_dict,target_keys)
    del state_dict
    tmpdir = tempfile.mkdtemp('temporary')
    try:
        mr.save_model(tmpdir)
        saved_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=tmpdir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        check_model(saved_model,target_keys)
        shutil.rmtree(tmpdir)
    except:
        shutil.rmtree(tmpdir)
        assert False
    del mr

# @pytest.mark.skip()
def test_ray_no_dp():
    target_keys = get_target_keys()
    # ray no dp
    cluster_address = "auto"
    config = Config.from_file("projects/ppo/internlm2/1B/actor_1gpu.py")
    coordinator = Coordinator(cluster_address, config)
    model_dict = coordinator.create_models()
    server:BaseModelServer = model_dict["actor"]
    model= server.model_get()
    check_model(model,target_keys)
    del model
    state_dict = server.state_dict_get()
    check_state_dict(state_dict,target_keys)
    del state_dict
    tmpdir = tempfile.mkdtemp('temporary')
    try:
        server.save_model(tmpdir)
        saved_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=tmpdir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        check_model(saved_model,target_keys)
        shutil.rmtree(tmpdir)
    except:
        shutil.rmtree(tmpdir)
        assert False
    coordinator.clean_up()

# @pytest.mark.skip()
def test_ray_ddp():
    target_keys = get_target_keys()
    # ray no dp
    cluster_address = "auto"
    config = Config.from_file("projects/ppo/internlm2/1B/actor_2gpu.py")
    coordinator = Coordinator(cluster_address, config)
    model_dict = coordinator.create_models()
    server:BaseModelServer = model_dict["actor"]
    model= server.model_get()
    check_model(model,target_keys)
    del model
    state_dict = server.state_dict_get()
    check_state_dict(state_dict,target_keys)
    del state_dict
    tmpdir = tempfile.mkdtemp('temporary')
    try:
        server.save_model(tmpdir)
        saved_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=tmpdir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        check_model(saved_model,target_keys)
        shutil.rmtree(tmpdir)
    except:
        shutil.rmtree(tmpdir)
        assert False
    coordinator.clean_up()

# @pytest.mark.skip()
def test_ray_fsdp():
    target_keys = get_target_keys()
    # ray no dp
    cluster_address = "auto"
    config = Config.from_file("projects/ppo/internlm2/1B/actor_2gpu_fsdp.py")
    coordinator = Coordinator(cluster_address, config)
    model_dict = coordinator.create_models()
    server:BaseModelServer = model_dict["actor"]
    state_dict = server.state_dict_get()
    check_state_dict(state_dict,target_keys)
    del state_dict
    tmpdir = tempfile.mkdtemp('temporary')
    try:
        server.save_model(tmpdir)
        saved_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=tmpdir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        check_model(saved_model,target_keys)
        shutil.rmtree(tmpdir)
    except:
        shutil.rmtree(tmpdir)
        assert False
    coordinator.clean_up()

# @pytest.mark.skip()
def test_ray_deepspeed():
    target_keys = get_target_keys()
    # ray no dp
    cluster_address = "auto"
    config = Config.from_file("projects/ppo/internlm2/1B/actor_2gpu_deepspeed.py")
    coordinator = Coordinator(cluster_address, config)
    model_dict = coordinator.create_models()
    server:BaseModelServer = model_dict["actor"]
    state_dict = server.state_dict_get()
    check_state_dict(state_dict,target_keys)
    del state_dict
    tmpdir = tempfile.mkdtemp('temporary')
    try:
        server.save_model(tmpdir)
        saved_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=tmpdir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        check_model(saved_model,target_keys)
        shutil.rmtree(tmpdir)
    except:
        shutil.rmtree(tmpdir)
        assert False
    coordinator.clean_up()


