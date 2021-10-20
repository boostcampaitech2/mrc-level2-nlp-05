from abc import abstractmethod
from typing import List

from arguments import (
    DatasetArguments
)
from tokenizers import Tokenizer

class Preprocessor:

    tokenizer = None
    column_names = None

    question_column = "question"
    context_column  = "context"
    answer_column   = "answers"

    def __init__(self, dataset_args: DatasetArguments, **kwargs):
        self.dataset_args = dataset_args

        if 'tokenizer' in kwargs:
            self.set_tokenizer(kwargs.get('tokenizer'))
        if 'column_names' in kwargs:
            self.set_column_names(kwargs.get('column_names'))

    def preprocess(self, examples, train:bool = True, **kwargs):
        # 만약 kwargs를 활용해야 한다면 override 해주세요! 

        if train:
            return self.prepare_train_features(examples)
        else:
            return self.prepare_valid_features(examples)

    @abstractmethod
    def prepare_train_features(self, examples, **kwargs):
        pass

    @abstractmethod
    def prepare_eval_features(self, examples, **kwargs):
        pass

    def get_tokenized_examples(self, examples):
        tokenized_examples = self.tokenizer(
            examples[self.question_column],
            examples[self.context_column],
            truncation="only_second",
            max_length=self.dataset_args.max_seq_len,
            stride=self.dataset_args.stride_len,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if self.dataset_args.use_max_padding else False,
        )
        return tokenized_examples

    def set_tokenizer(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def set_column_names(self, column_names):
        self.column_names = column_names
        self.question_column = "question" if "question" in column_names else column_names[0]
        self.context_column  = "context"  if "context"  in column_names else column_names[1]
        self.answer_column   = "answers"  if "answers"  in column_names else column_names[2]


class BaselinePreprocessor(Preprocessor):

    def prepare_train_features(self, examples):

        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, 
        # stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.

        # pad_on_right == True라고 가정하고 작성했습니다.

        # TODO: tokenizer roberta일 경우 token_type_ids 반환하지 않도록 initialization시 설정
        tokenized_examples = self.get_tokenized_examples(examples)

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
        # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # 데이터셋에 "start position", "enc position" label을 부여합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)  # cls index

            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            answers = examples[self.answer_column][sample_index]

            # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # text에서 정답의 Start/end character index
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # text에서 current span의 Start token index
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # text에서 current span의 End token index
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                    # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_eval_features(self, examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = self.get_tokenized_examples(examples)

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples


