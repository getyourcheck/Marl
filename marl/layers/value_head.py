import torch


class ValueHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.pooled = torch.nn.Linear(
            self.hidden_size, self.hidden_size, bias_attr=config.use_bias
        )
        self.score = torch.nn.Linear(self.hidden_size, 1, bias_attr=config.use_bias)

    def forward(self, hidden_states):
        hidden_states = torch.nn.functional.tanh(self.pooled(hidden_states))
        logits = self.score(hidden_states)
        return logits
