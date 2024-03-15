import torch


def get_question_answer_mask(
    input_ids: torch.Tensor,
    output_ids: torch.Tensor,
    tokenizer_pad_token_id: int,
    generate_pad_token_id: int = None,
):
    """
    Example:
        input_ids = torch.tensor([[0, 1, 9]])
        output_ids = torch.tensor([[0, 1, 9, 2, 3, 4, 5]])
        tokenizer_pad_token_id = 0  # set 0 as neither question or answer
        generate_pad_token_id = None
        expected_qst_mask = torch.tensor([[0, 1, 1, 0, 0, 0, 0]])
        expected_ans_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    """
    # seq_mask yields zero where token == pad_token_id
    seq_mask = output_ids.not_equal(tokenizer_pad_token_id).int()
    if generate_pad_token_id is not None:
        seq_mask *= output_ids.not_equal(generate_pad_token_id).int()

    question_len = input_ids.shape[-1]
    question_mask = seq_mask.clone()
    question_mask[:, question_len:] = 0
    answer_mask = seq_mask.clone()
    answer_mask[:, :question_len] = 0
    return question_mask, answer_mask
