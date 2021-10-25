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