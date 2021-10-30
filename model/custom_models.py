import sys
sys.path.append('..')

from transformers.models.roberta.modeling_roberta import RobertaForQuestionAnswering
from model.custom_heads import *

class CustomRobertaForQuestionAnsweringWithCNNHead(RobertaForQuestionAnswering):
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithCNNHead, self).__init__(config)
        self.qa_outputs = CustomHeadCNNWithDropout(config)

class CustomRobertaForQuestionAnsweringWithLSTMCNNHead(RobertaForQuestionAnswering):
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithLSTMCNNHead, self).__init__(config)
        self.qa_outputs = CustomHeadLSTMCNN(config)

class CustomRobertaForQuestionAnsweringWithAttentionHead(RobertaForQuestionAnswering):
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithAttentionHead, self).__init__(config)
        self.qa_outputs = CustomHeadAttention(config)
    
class CustomRobertaForQuestionAnsweringWithAttentionCNNHead(RobertaForQuestionAnswering):
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithAttentionCNNHead, self).__init__(config)
        self.qa_outputs = CustomHeadAttentionCNN(config)

class CustomRobertaForQuestionAnsweringWithCNNAttentionHead(RobertaForQuestionAnswering):
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithCNNAttentionHead, self).__init__(config)
        self.qa_outputs = CustomHeadCNNAttention(config)

class CustomRobertaForQuestionAnsweringWithMultiHeadAttentionHead(RobertaForQuestionAnswering):
    def __init__(self, config):
        super(CustomRobertaForQuestionAnsweringWithMultiHeadAttentionHead, self).__init__(config)
        self.qa_outputs = CustomHeadMultiHeadAttention(config)