import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


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

class CustomHeadCNN(nn.Module):
    """
    3개의 convolution layer를 이용하여 차원을 압축한 결과를 concat 한 뒤 fc layer를 통과시킴
    """
    def __init__(self, config):
        super(CustomHeadCNN, self).__init__()
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3,
            kernel_size=1, 
            padding=0)  # stride: default 1
        self.conv_3 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3, 
            kernel_size=3, 
            padding=1)
        self.conv_5 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3 + 1,  # concat 합칠 때 맞아 떨어지도록
            kernel_size=5, 
            padding=2)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        conv1_out = self.relu(self.conv_1(x).transpose(1, 2).contiguous())
        conv3_out = self.relu(self.conv_3(x).transpose(1, 2).contiguous())
        conv5_out = self.relu(self.conv_5(x).transpose(1, 2).contiguous())
        output = self.fc(torch.cat((conv1_out, conv3_out, conv5_out), -1)) # concat
        return output

class CustomHeadCNNWithDropout(nn.Module):
    """
    3개의 convolution layer를 이용하여 차원을 압축한 결과를 concat 한 뒤 dropout을 적용하여 fc layer를 통과시킴
    """
    def __init__(self, config):
        super(CustomHeadCNNWithDropout, self).__init__()
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3,
            kernel_size=1, 
            padding=0)  # stride: default 1
        self.conv_3 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3, 
            kernel_size=3, 
            padding=1)
        self.conv_5 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3 + 1,  # concat 합칠 때 맞아 떨어지도록
            kernel_size=5, 
            padding=2)
        self.dropout = nn.Dropout(p=config.dropout_ratio)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        conv1_out = self.relu(self.conv_1(x).transpose(1, 2).contiguous())
        conv3_out = self.relu(self.conv_3(x).transpose(1, 2).contiguous())
        conv5_out = self.relu(self.conv_5(x).transpose(1, 2).contiguous())
        output = self.fc(self.dropout(torch.cat((conv1_out, conv3_out, conv5_out), -1))) # concat, dropout
        return output

class CustomHeadCNNWithMaxPool(nn.Module):
    def __init__(self, config):
        super(CustomHeadCNNWithMaxPool, self).__init__()
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3,
            kernel_size=1, 
            padding=0)  # stride: default 1
        self.conv_3 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3, 
            kernel_size=3, 
            padding=1)
        self.conv_5 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3 + 1,  # concat 합칠 때 맞아 떨어지도록
            kernel_size=5, 
            padding=2)
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.dropout = nn.Dropout(p=config.dropout_ratio)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = x.transpose(1,2).contiguous()
        conv1_out = self.maxpool(self.relu(self.conv_1(x).transpose(1, 2).contiguous()))
        conv3_out = self.maxpool(self.relu(self.conv_3(x).transpose(1, 2).contiguous()))
        conv5_out = self.maxpool(self.relu(self.conv_5(x).transpose(1, 2).contiguous()))
        output = self.fc(self.dropout(torch.cat((conv1_out, conv3_out, conv5_out), -1)))
        return output

class CustomHeadLSTM(nn.Module):
    """
    양방향 LSTM 2 layer를 사용한 뒤 dropout을 적용하여 fc layer를 통과시킴
    """
    def __init__(self, config):
        super(CustomHeadLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=2,                   # layer수 변화에 따른 차원의 변화는 없음
            dropout=config.dropout_ratio,
            bidirectional=True,             # 양방향 사용 시 벡터의 차원이 2배 늘어남
            batch_first=True
        )
        self.dropout = nn.Dropout(p=config.dropout_ratio)
        self.fc = nn.Linear(config.hidden_size*2, config.num_labels)

    def forward(self, x):
        output, (_, _) = self.lstm(x)
        output = self.fc(self.dropout(output))
        return output

class CustomHeadLSTMCNN(nn.Module):
    """
    양방향 LSTM 2 layer와 CNN을 결합
    """
    def __init__(self, config):
        super(CustomHeadLSTMCNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=2,                   # layer수 변화에 따른 차원의 변화는 없음
            dropout=config.dropout_ratio,
            bidirectional=True,             # 양방향 사용 시 벡터의 차원이 2배 늘어남
            batch_first=True
        )
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv1d(
            in_channels=config.hidden_size*2, 
            out_channels=config.hidden_size // 3,
            kernel_size=1, 
            padding=0)  # stride: default 1
        self.conv_3 = nn.Conv1d(
            in_channels=config.hidden_size*2, 
            out_channels=config.hidden_size // 3, 
            kernel_size=3, 
            padding=1)
        self.conv_5 = nn.Conv1d(
            in_channels=config.hidden_size*2, 
            out_channels=config.hidden_size // 3 + 1,  # concat 합칠 때 맞아 떨어지도록
            kernel_size=5, 
            padding=2)        
        self.dropout = nn.Dropout(p=config.dropout_ratio)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        output, (_, _) = self.lstm(x)
        output = output.transpose(1,2).contiguous()
        conv1_out = self.relu(self.conv_1(output).transpose(1, 2).contiguous())
        conv3_out = self.relu(self.conv_3(output).transpose(1, 2).contiguous())
        conv5_out = self.relu(self.conv_5(output).transpose(1, 2).contiguous())
        output = self.fc(self.dropout(torch.cat((conv1_out, conv3_out, conv5_out), -1)))
        return output

class CustomHeadLSTM_L3_CNN(nn.Module):
    """
    양방향 LSTM 3 layer와 CNN을 결합
    """
    def __init__(self, config):
        super(CustomHeadLSTM_L3_CNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=3,                   # layer수 변화에 따른 차원의 변화는 없음
            dropout=config.dropout_ratio,
            bidirectional=True,             # 양방향 사용 시 벡터의 차원이 2배 늘어남
            batch_first=True
        )
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv1d(
            in_channels=config.hidden_size*2, 
            out_channels=config.hidden_size // 3,
            kernel_size=1, 
            padding=0)  # stride: default 1
        self.conv_3 = nn.Conv1d(
            in_channels=config.hidden_size*2, 
            out_channels=config.hidden_size // 3, 
            kernel_size=3, 
            padding=1)
        self.conv_5 = nn.Conv1d(
            in_channels=config.hidden_size*2, 
            out_channels=config.hidden_size // 3 + 1,  # concat 합칠 때 맞아 떨어지도록
            kernel_size=5, 
            padding=2)        
        self.dropout = nn.Dropout(p=config.dropout_ratio)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        output, (_, _) = self.lstm(x)
        output = output.transpose(1,2).contiguous()
        conv1_out = self.relu(self.conv_1(output).transpose(1, 2).contiguous())
        conv3_out = self.relu(self.conv_3(output).transpose(1, 2).contiguous())
        conv5_out = self.relu(self.conv_5(output).transpose(1, 2).contiguous())
        output = self.fc(self.dropout(torch.cat((conv1_out, conv3_out, conv5_out), -1)))
        return output

    


class CustomHeadAttention(nn.Module):
    def __init__(self, config ):
        super().__init__() 
        self.config = config
        self.query_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.gelu = nn.GELU()
        self.classify_layer = nn.Linear(config.hidden_size, 2, bias=True)
        self.drop_out = nn.Dropout(0.5)

    def forward(self, x):
        sequence_output = x 
        dim_size = x.shape[2]

        embedded_query = self.query_layer(sequence_output)
        embedded_key = self.key_layer(sequence_output) 
        embedded_value = self.value_layer(sequence_output)

        attention_score = torch.matmul(embedded_query, torch.transpose(embedded_key, -2, -1)) / math.sqrt(dim_size)
        attention_dists = F.softmax(attention_score, dim=-1)
        logits = torch.matmul(attention_dists, embedded_value) 
        logits = self.gelu(logits)
        logits = self.drop_out(logits)
        logits = self.classify_layer(logits) 

        return logits

class CustomHeadAttentionWithLN(nn.Module):
    def __init__(self, config ):
        super().__init__() 
        self.config = config
        self.query_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.gelu = nn.GELU()
        self.classify_layer = nn.Linear(config.hidden_size, 2, bias=True)
        self.drop_out = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        sequence_output = x 
        dim_size = x.shape[2]

        embedded_query = self.query_layer(sequence_output)
        embedded_key = self.key_layer(sequence_output) 
        embedded_value = self.value_layer(sequence_output)

        attention_score = torch.matmul(embedded_query, torch.transpose(embedded_key, -2, -1)) / math.sqrt(dim_size)
        attention_dists = F.softmax(attention_score, dim=-1)
        logits = torch.matmul(attention_dists, embedded_value) 
        logits = self.gelu(logits)
        logits = self.drop_out(logits)
        logits = self.layer_norm(x + logits)
        logits = self.classify_layer(logits) 

        return logits


class CustomHeadAttention_V2(nn.Module):
    def __init__(self, config):
        super(CustomHeadAttention_V2, self).__init__()

        self.query_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.dropout_ratio)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states):
        query = self.query_layer(hidden_states)
        key = self.key_layer(hidden_states)
        value = self.value_layer(hidden_states)

        d_k = key.shape[-1]

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn_dists = F.softmax(attn_scores, dim=-1)
        attn_values = torch.matmul(attn_dists, value)

        logit = self.fc(self.dropout(attn_values))

        return logit

class CustomHeadAttentionWithLN_V2(nn.Module):
    def __init__(self, config):
        super(CustomHeadAttentionWithLN_V2, self).__init__()
        self.query_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.dropout_ratio)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states):
        query = self.query_layer(hidden_states)
        key = self.key_layer(hidden_states)
        value = self.value_layer(hidden_states)

        d_k = key.shape[-1]

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn_dists = F.softmax(attn_scores, dim=-1)
        attn_values = torch.matmul(attn_dists, value)

        output = self.layer_norm(hidden_states + self.dropout(attn_values))
        logit = self.fc(output)

        return logit

class CustomHeadAttentionCNN(nn.Module):
    def __init__(self, config ):
        super().__init__() 
        self.config = config
        self.query_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.gelu = nn.GELU()
        self.classify_layer = nn.Linear(config.hidden_size, 2, bias=True)
        self.drop_out = nn.Dropout(0.5)


        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3,
            kernel_size=1, 
            padding=0)  # stride: default 1
        self.conv_3 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3, 
            kernel_size=3, 
            padding=1)
        self.conv_5 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3, 
            kernel_size=5, 
            padding=2)
        self.fc = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        sequence_output = x
        dim_size = x.shape[2]
        embedded_query = self.query_layer(sequence_output) 
        embedded_key = self.key_layer(sequence_output) 
        embedded_value = self.value_layer(sequence_output) 

        attention_score = torch.matmul(embedded_query, torch.transpose(embedded_key, -2, -1)) / math.sqrt(dim_size)
        attention_dists = F.softmax(attention_score, dim=-1)
        logits = torch.matmul(attention_dists, embedded_value) 
        logits = self.gelu(logits)
        logits = self.drop_out(logits)
        x = logits.transpose(1, 2).contiguous()
        conv1_out = self.relu(self.conv_1(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3_out = self.relu(self.conv_3(x).transpose(1, 2).contiguous().squeeze(-1))
        conv5_out = self.relu(self.conv_5(x).transpose(1, 2).contiguous().squeeze(-1))
        output = self.fc(torch.cat((conv1_out, conv3_out, conv5_out), -1))

        return output

class CustomHeadCNNAttention(nn.Module):
    def __init__(self, config ):
        super().__init__() 
        self.config = config
        self.query_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.gelu = nn.GELU()
        self.classify_layer = nn.Linear(config.hidden_size, 2, bias=True)
        self.drop_out = nn.Dropout(0.5)


        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3,
            kernel_size=1, 
            padding=0)  # stride: default 1
        self.conv_3 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3, 
            kernel_size=3, 
            padding=1)
        self.conv_5 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.hidden_size // 3, 
            kernel_size=5, 
            padding=2)
        self.fc = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        conv1_out = self.relu(self.conv_1(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3_out = self.relu(self.conv_3(x).transpose(1, 2).contiguous().squeeze(-1))
        conv5_out = self.relu(self.conv_5(x).transpose(1, 2).contiguous().squeeze(-1))
        x = torch.cat((conv1_out, conv3_out, conv5_out), -1)
        sequence_output = x
        dim_size = x.shape[2]
        embedded_query = self.query_layer(sequence_output) 
        embedded_key = self.key_layer(sequence_output) 
        embedded_value = self.value_layer(sequence_output) 

        attention_score = torch.matmul(embedded_query, torch.transpose(embedded_key, -2, -1)) / math.sqrt(dim_size)
        attention_dists = F.softmax(attention_score, dim=-1)
        logits = torch.matmul(attention_dists, embedded_value) 
        logits = self.gelu(logits)
        logits = self.drop_out(logits)
        logits = self.classify_layer(logits) 
        return logits

class CustomHeadMultiHeadAttention(nn.Module):
    def __init__(self, config ):
        super().__init__() 
        self.config = config
        self.num_heads = 8
        self.d_k = config.hidden_size // self.num_heads
        self.query_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.gelu = nn.GELU()
        self.classify_layer = nn.Linear(config.hidden_size, 2, bias=True)
        self.drop_out = nn.Dropout(0.5)

    def forward(self, x):
        sequence_output = x
        batch_size = x.shape[0]

        embedded_query = self.query_layer(sequence_output)
        embedded_key = self.key_layer(sequence_output) 
        embedded_value = self.value_layer(sequence_output) 

        embedded_query = embedded_query.view(batch_size, -1, self.num_heads, self.d_k) 
        embedded_key = embedded_key.view(batch_size, -1, self.num_heads, self.d_k)  
        embedded_value = embedded_value.view(batch_size, -1, self.num_heads, self.d_k)  

        embedded_query = embedded_query.transpose(1, 2) 
        embedded_key = embedded_key.transpose(1, 2)  
        embedded_value = embedded_value.transpose(1, 2)

        attention_score = torch.matmul(embedded_query, torch.transpose(embedded_key, -2, -1)) / math.sqrt(self.d_k)
        attention_dists = F.softmax(attention_score, dim=-1) 
        attention_value = torch.matmul(attention_dists, embedded_value)
        logits = attention_value.transpose(1, 2).contiguous().view(batch_size, -1, self.config.hidden_size) 
        logits = self.gelu(logits)
        logits = self.drop_out(logits)
        logits = self.classify_layer(logits) 

        return logits
