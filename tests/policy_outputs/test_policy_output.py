import torch
from marl.policy_output import (
    PolicyOutput,
    concat_policy_outputs,
    padding_policy_outputs,
)


def test_concat_policy_outputs():
    a = PolicyOutput(
        output_ids=torch.ones(2),
        output_str=["Hello World"],
    )

    b = PolicyOutput(
        output_ids=torch.zeros(3),
        output_str=["Large Language Model"],
    )

    x = PolicyOutput(
        output_ids=torch.cat([torch.ones(2), torch.zeros(3)]),
        output_str=["Hello World", "Large Language Model"],
    )
    y = concat_policy_outputs([a, b], padding_token_map=None)  # disable padding
    assert x == y, f"\nx={x}\ny={y}"


def test_padding_policy_outputs():
    PADDING_ID = 10
    MAX_SEQ_LEN = 192
    a = PolicyOutput(output_ids=torch.ones(32, 128))
    b = PolicyOutput(output_ids=torch.ones(18, MAX_SEQ_LEN))
    c = PolicyOutput(output_ids=torch.ones(64, 10))
    policy_outputs = [a, b, c]
    padding_token_map = {"output_ids": PADDING_ID}
    policy_outputs = padding_policy_outputs(policy_outputs, padding_token_map)
    assert policy_outputs[0]["output_ids"].shape == torch.Size([32, MAX_SEQ_LEN])
    assert policy_outputs[1]["output_ids"].shape == torch.Size([18, MAX_SEQ_LEN])
    assert policy_outputs[2]["output_ids"][-1][MAX_SEQ_LEN - 1] == PADDING_ID


def test_concat_outout_ids():
    PADDING_ID = 10
    MAX_SEQ_LEN = 192
    a = PolicyOutput(output_ids=torch.ones(32, 128))
    b = PolicyOutput(output_ids=torch.ones(18, MAX_SEQ_LEN))
    c = PolicyOutput(output_ids=torch.ones(64, 10))
    padding_token_map = {"output_ids": PADDING_ID}
    y = concat_policy_outputs([a, b, c], padding_token_map=padding_token_map)
    assert y["output_ids"].shape == torch.Size([32 + 18 + 64, MAX_SEQ_LEN])
    assert y["output_ids"][-1][-1] == PADDING_ID


if __name__ == "__main__":
    test_concat_policy_outputs()
    test_padding_policy_outputs()
    test_concat_outout_ids()
