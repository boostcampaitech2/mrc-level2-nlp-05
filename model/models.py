import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import RobertaForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.utils.dummy_pt_objects import RobertaForQuestionAnswering


class MeanPooler(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_p)

    def forward(self, hidden_states, attention_mask, sqrt=True):
        # hidden states: [batch_size, seq, model_dim]
        # attention masks: [batch_size, seq, 1]

        sentence_sums = torch.bmm(hidden_states.permute(0, 2, 1), attention_mask.float().unsqueeze(-1)).squeeze(-1)
        divisor = attention_mask.sum(dim=1).view(-1, 1).float()

        if sqrt:
            divisor = divisor.sqrt()
        sentence_sums /= divisor

        pooled_output = self.dense1(sentence_sums)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.dense2(pooled_output)

        return pooled_output


class CustomQAOutputLinear(nn.Module):

    def __init__(self, config):
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        return self.qa_outputs(x)


class CustomQAOutputRNN(nn.Module):

    def __init__(self, config):
        super(CustomQAOutputRNN, self).__init__()
        
        if config.rnn_type == "GRU":
            self.rnn_p1 = nn.GRU(
                config.hidden_size, config.hidden_size,
                num_layers=1, bidirectional=True, batch_first=True
            )
            self.rnn_p2 = nn.GRU(
                config.hidden_size, config.hidden_size,
                num_layers=1, bidirectional=True, batch_first=True
            )
        else:
            self.rnn_p1 = nn.LSTM(
                config.hidden_size, config.hidden_size,
                num_layers=1, bidirectional=True, batch_first=True
            )
            self.rnn_p2 = nn.LSTM(
                config.hidden_size, config.hidden_size,
                num_layers=1, bidirectional=True, batch_first=True
            )


class CustomRobertaForQuestionAnsweringRNN(RobertaForQuestionAnswering):
    pass
        
        
