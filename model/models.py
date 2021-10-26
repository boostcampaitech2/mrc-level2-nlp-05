import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, num_labels: int):
        super(BaseModel, self).__init__()

        if num_labels < 1:
            num_labels = 1
        
        self.fc = nn.Linear(16, num_labels)
        
    def forward(self, x):
        return F.relu(self.fc(x))

class CustomHeadBase(nn.Module):
    def __init__(self, config):
        super(CustomHeadBase, self).__init__()
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, hidden_states):
        # hidden_states: (batch_size, max_seq_len, backbone_model_output_dim)
        return self.qa_outputs(hidden_states)


class CustomHeadRNN(nn.Module):
    def __init__(self, config):
        super(CustomHeadRNN, self).__init__()
        
        self.p1_lstm = nn.LSTM(
            config.hidden_size,config.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            # proj_size=config.hidden_size // 4,
            dropout=config.dropout_ratio)
        self.p1_drop = nn.Dropout(p=config.dropout_ratio)
        self.p1_fc   = nn.Linear(2*config.hidden_size, 1)

        self.p2_lstm = nn.LSTM(
            config.hidden_size,
            config.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            # proj_size=config.hidden_size // 4,
            dropout=config.dropout_ratio)
        self.p2_drop = nn.Dropout(p=config.dropout_ratio)
        self.p2_fc   = nn.Linear(2*config.hidden_size, 1)

    def forward(self, hidden_states):
        p1_out, (h_p1, c_p1) = self.p1_lstm(hidden_states)
        p1_out = self.p1_drop(p1_out)
        p1_out = self.p1_fc(p1_out)  # start logit

        p2_out, (h_p2, c_p2) = self.p2_lstm(hidden_states, (h_p1, c_p1))
        p2_out = self.p2_drop(p2_out)
        p2_out = self.p2_fc(p2_out)  # end logit

        return torch.cat([p1_out, p2_out], dim=-1)


