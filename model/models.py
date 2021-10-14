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