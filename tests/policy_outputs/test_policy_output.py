import torch
from marl.policy_output import PolicyOutput, concat_policy_outputs


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

    y = concat_policy_outputs([a, b])
    assert x == y, f"\nx={x}\ny={y}"


if __name__ == "__main__":
    test_concat_policy_outputs()
