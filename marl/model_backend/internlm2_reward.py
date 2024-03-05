# adapted from https://gitlab.pjlab.org.cn/openmmlab/bigmodel/rl3m/-/blob/main/scripts/transformers/internlm2_model/modeling_internlm2.py#L1271
from typing import Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_outputs import (
    SequenceClassifierOutputWithPast,
)
from transformers.utils import (
    replace_return_docstrings,
)

InternLM2PreTrainedModel = get_class_from_dynamic_module(
    class_reference="internlm/internlm2-chat-1_8b-sft--modeling_internlm2.InternLM2PreTrainedModel",
    pretrained_model_name_or_path="internlm/internlm2-chat-1_8b-sft",
)
InternLM2Model = get_class_from_dynamic_module(
    class_reference="internlm/internlm2-chat-1_8b-sft--modeling_internlm2.InternLM2Model",
    pretrained_model_name_or_path="internlm/internlm2-chat-1_8b-sft",
)


class InternLM2ForRewardModel(InternLM2PreTrainedModel):

    _tied_weights_keys = ["v_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.reward_token_id = config.reward_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.v_head

    def set_output_embeddings(self, new_embeddings):
        self.v_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @replace_return_docstrings(
        output_type=SequenceClassifierOutputWithPast, config_class="InternLM2Config"
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SequenceClassifierOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, InternLM2ForCausalLM

        >>> model = InternLM2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # make sure that the last token is the reward token
        assert (
            input_ids[:, -1].eq(self.reward_token_id).all().item()
        ), "The last token of input_ids mast be the reward token"

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        reward_scores = self.v_head(hidden_states)
        # get last token's reward score
        reward_scores = reward_scores[:, -1].view(-1, 1)

        loss = None

        if not return_dict:
            output = (reward_scores,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=reward_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def get_score(
        self,
        tokenizer,
        conversation: list[dict],
        **kwargs,
    ):
        conversation_str = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        input_ids = tokenizer.encode(conversation_str, return_tensors="pt")
        # add reward score token at the end of the input_ids
        input_ids = torch.cat(
            [input_ids, torch.tensor([[self.reward_token_id]], dtype=torch.long)], dim=1
        ).to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool).to(self.device)

        outputs = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        score = outputs[0].cpu().item()
        return score

    @torch.no_grad()
    def get_scores(
        self,
        tokenizer,
        conversations: list[list[dict]],
        **kwargs,
    ):
        conversation_strs = [
            tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            for conversation in conversations
        ]
        input_ids = tokenizer.batch_encode_plus(
            conversation_strs, return_tensors="pt", padding=True
        )
        # add reward score token at the end of the input_ids
        input_ids["input_ids"] = torch.cat(
            [
                input_ids["input_ids"],
                torch.tensor([[self.reward_token_id]], dtype=torch.long).expand(
                    input_ids["input_ids"].shape[0], 1
                ),
            ],
            dim=1,
        ).to(self.device)
        attention_mask = torch.cat(
            [
                input_ids["attention_mask"],
                torch.ones(input_ids["attention_mask"].shape[0], 1, dtype=torch.bool),
            ],
            dim=1,
        ).to(self.device)

        outputs = self.forward(
            input_ids=input_ids["input_ids"], attention_mask=attention_mask, **kwargs
        )
        scores = outputs[0].cpu().tolist()
        return scores

    @torch.no_grad()
    def compare(
        self,
        tokenizer,
        conversation1: list[dict],
        conversation2: list[dict],
        return_logits: bool = False,
        **kwargs,
    ):
        score1 = self.get_score(tokenizer, conversation1, **kwargs)
        score2 = self.get_score(tokenizer, conversation2, **kwargs)
        if return_logits:
            return score1, score2
        else:
            return score1 > score2

    @torch.no_grad()
    def rank(
        self,
        tokenizer,
        conversations: list[list[dict]],
        return_logits: bool = False,
        **kwargs,
    ):
        scores = self.get_scores(tokenizer, conversations, **kwargs)
        if return_logits:
            return scores
        else:
            return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
