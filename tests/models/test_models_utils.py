import torch
import unittest
from marl.model_backend.generate_utils import get_question_answer_mask


class TestQuestionAnswerMask(unittest.TestCase):
    def test_get_question_answer_mask_batch_1(self):
        input_ids = torch.tensor([[0, 1, 9]])
        output_ids = torch.tensor([[0, 1, 9, 2, 3, 4, 5]])
        tokenizer_pad_token_id = 0  # set 0 as neither question or answer
        generate_pad_token_id = None

        qst_mask, ans_mask = get_question_answer_mask(
            input_ids, output_ids, tokenizer_pad_token_id, generate_pad_token_id
        )
        print(qst_mask, ans_mask)
        expected_qst_mask = torch.tensor([[0, 1, 1, 0, 0, 0, 0]])
        expected_ans_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
        self.assertTrue(torch.all(torch.eq(qst_mask, expected_qst_mask)))
        self.assertTrue(torch.all(torch.eq(ans_mask, expected_ans_mask)))

    def test_get_question_answer_mask_batch_2(self):
        input_ids = torch.tensor([[0, 1, 2, 9], [0, 1, 9, 9]])
        output_ids = torch.tensor([[0, 1, 2, 9, 4, 5, 6], [0, 1, 9, 9, 2, 3, 4]])
        tokenizer_pad_token_id = 0  # set 0 as neither question or answer
        generate_pad_token_id = 9  # set 9 as neither question or answer

        qst_mask, ans_mask = get_question_answer_mask(
            input_ids, output_ids, tokenizer_pad_token_id, generate_pad_token_id
        )
        print(qst_mask, ans_mask)
        expected_qst_mask = torch.tensor([[0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]])
        expected_ans_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1]])
        self.assertTrue(torch.all(torch.eq(qst_mask, expected_qst_mask)))
        self.assertTrue(torch.all(torch.eq(ans_mask, expected_ans_mask)))
