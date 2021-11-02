import sys
sys.path.append('..')

from transformers.models.roberta.modeling_roberta import RobertaForQuestionAnswering
from model.custom_heads import *

class CustomRobertaForQuestionAnsweringWithRNNHead(RobertaForQuestionAnswering):
    """RoBERTa QA Model with RNN Custom Head"""
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithRNNHead, self).__init__(config)
        self.qa_outputs = CustomHeadRNN(config)

class CustomRobertaForQuestionAnsweringWithCNNHead(RobertaForQuestionAnswering):
    """RoBERTa QA Model with CNN Custom Head"""
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithCNNHead, self).__init__(config)
        self.qa_outputs = CustomHeadCNNWithDropout(config)

class CustomRobertaForQuestionAnsweringWithLSTMCNNHead(RobertaForQuestionAnswering):
    """RoBERTa QA Model with LSTM + CNN Custom Head"""
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithLSTMCNNHead, self).__init__(config)
        self.qa_outputs = CustomHeadLSTMCNN(config)

class CustomRobertaForQuestionAnsweringWithAttentionHead(RobertaForQuestionAnswering):
    """RoBERTa QA Model with Attention Custom Head (by zeu)"""
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithAttentionHead, self).__init__(config)
        self.qa_outputs = CustomHeadAttention(config)

class CustomRobertaForQuestionAnsweringWithAttentionWithLNHead(RobertaForQuestionAnswering):
    """RoBERTa QA Model with Attention + LayerNorm Custom Head (by zeu)"""
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithAttentionWithLNHead, self).__init__(config)
        self.qa_outputs = CustomHeadAttentionWithLN(config)

class CustomRobertaForQuestionAnsweringWithAttentionHead_V2(RobertaForQuestionAnswering):
    """RoBERTa QA Model with Attention Custom Head (by sunghan)"""
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithAttentionHead_V2, self).__init__(config)
        self.qa_outputs = CustomHeadAttention_V2(config)

class CustomRobertaForQuestionAnsweringWithAttentionWithLNHead_V2(RobertaForQuestionAnswering):
    """RoBERTa QA Model with Attention + LayerNorm Custom Head (by sunghan)"""
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithAttentionWithLNHead_V2, self).__init__(config)
        self.qa_outputs = CustomHeadAttentionWithLN_V2(config)    
    
class CustomRobertaForQuestionAnsweringWithAttentionCNNHead(RobertaForQuestionAnswering):
    """RoBERTa QA Model with Attention + CNN Custom Head"""
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithAttentionCNNHead, self).__init__(config)
        self.qa_outputs = CustomHeadAttentionCNN(config)

class CustomRobertaForQuestionAnsweringWithCNNAttentionHead(RobertaForQuestionAnswering):
    """RoBERTa QA Model with CNN + Attention Custom Head"""
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithCNNAttentionHead, self).__init__(config)
        self.qa_outputs = CustomHeadCNNAttention(config)

class CustomRobertaForQuestionAnsweringWithMultiHeadAttentionHead(RobertaForQuestionAnswering):
    """RoBERTa QA Model with Multi-Head Attention Custom Head"""
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithMultiHeadAttentionHead, self).__init__(config)
        self.qa_outputs = CustomHeadMultiHeadAttention(config)