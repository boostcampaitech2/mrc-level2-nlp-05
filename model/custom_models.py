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