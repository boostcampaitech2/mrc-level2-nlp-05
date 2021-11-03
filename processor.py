import json
import os
import random
import logging

from collections import defaultdict, OrderedDict
from typing import List, Tuple, Optional, Union
from datasets.utils.deprecation_utils import deprecated

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import TensorDataset

from tqdm import tqdm

from transformers.data.processors.utils import DataProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import PreTrainedTokenizer, EvalPrediction

from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers.training_args import TrainingArguments

from arguments import DatasetArguments


logger = logging.getLogger(__name__)


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def softmax(predictions, column_name):
    scores = np.array([pred.pop(column_name) for pred in predictions])
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    return probs


class QAProcessor(DataProcessor):

    def __init__(
        self, 
        dataset_args: DatasetArguments,
        tokenizer: PreTrainedTokenizerBase, 
        concat: bool = False,
        **kwargs
    ):
        """
        Args:
            data_dir: str
                Directory of Datasets' arrow dataset
                Default set ot "/opt/ml/data"
            tokenizer: Tokenizer
            concat: bool
                Concatenate both train and validation set if set to True. 
                Default: False
        """

        self.dataset_args = dataset_args

        if dataset_args.dataset_path is None:
            self.data_dir = "/opt/ml/data"
        else:
            self.data_dir = dataset_args.dataset_path

        self.train_datasets_dir = os.path.join(self.data_dir, "train_dataset")
        self.test_datasets_dir  = os.path.join(self.data_dir, "test_dataset")

        self.train_datasets = load_from_disk(self.train_datasets_dir)
        self.datasets = self.train_datasets
        self.test_datasets  = load_from_disk(self.test_datasets_dir)

        self.train_dataset = self.train_datasets["train"]
        self.eval_dataset  = self.train_datasets["validation"]
        self.test_dataset  = self.test_datasets["validation"]

        self.tokenizer = tokenizer

        self.is_train = True

        if concat:
            # concatenate train and eval set to train set
            self.train_dataset = concatenate_datasets([self.train_dataset, self.eval_dataset])

        # self.train_dataset = self.train_dataset.map(self._flatten_multiple_answers, batched=True, batch_size=1, remove_columns=["answers"])
        # self.eval_dataset  = self.eval_dataset.map(self._flatten_multiple_answers, batched=True, batch_size=1, remove_columns=["answers"])

        self.set_column_names()


    def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    
    def set_column_names(self, column_names: List[str] = None):
        """Set column names for processing data.
        
        Args:
            column_names: `[question_column, context_column, answer_column]`
        """
        default_names = ["question", "context", "answers"]
        self.column_names    = column_names if column_names is not None else default_names

        self.question_column = column_names[0] if column_names is not None else default_names[0]
        self.context_column  = column_names[1] if column_names is not None else default_names[1]
        self.answer_column   = column_names[2] if column_names is not None else default_names[2]
    

    def _flatten_multiple_answers(self, examples):
        """Now this processor can handle multiple answers and split those into seperate rows by default"""

        results = defaultdict(list)

        column_names = list(examples.keys())
        column_names.remove("answers")

        answers = examples['answers'][0]
        start_positions = answers['answer_start']
        answer_texts = answers['text']

        if len(start_positions) == 0:
            # which means there is no answer in the context (negative samples)
            results['start_position'].append(0)
            results['end_position'].append(0)
            results['answer_text'].append("")
            results['is_impossible'].append(True)

            for column_name in column_names:
                results[column_name].append(examples[column_name][0])

        else:

            for start_position, answer_text in zip(start_positions, answer_texts):
                results['start_position'].append(start_position)
                results['answer_text'].append(answer_text)
                results['end_position'].append(start_position + len(answer_text) - 1)
                results['is_impossible'].append(False)

                for column_name in column_names:
                    results[column_name].append(examples[column_name][0])

        return results

    
    def _add_title_to_context(self, example):
        
        prefix = example['title'] + ": "

        example['context'] = prefix + example['context']
        example['start_position'] += len(prefix)
        example['end_position'] += len(prefix)

        return example


    def add_title_to_contexts(self, to_train: bool = False, to_eval: bool = False):

        if not to_train and not to_eval:
            raise ValueError("Nothing happened")
        
        # test dataset does not have contexts
        if to_train and self.train_dataset:
            self.train_dataset = self.train_dataset.map(self._add_title_to_context, batched=False)
        if to_eval and self.eval_dataset:
            self.eval_dataset = self.eval_dataset.map(self._add_title_to_context, batched=False)


    def get_tokenized_features(self, examples):

        pad_on_right = self.tokenizer.padding_side == "right"

        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        features = self.tokenizer(
            examples[self.question_column if pad_on_right else self.context_column],
            examples[self.context_column  if pad_on_right else self.question_column],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.dataset_args.max_seq_len,
            stride=self.dataset_args.stride_len,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            padding="max_length",
        )

        # delete token_type_ids from featrues
        MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}

        tokenizer_type = type(self.tokenizer).__name__.replace("Tokenizer", "").lower()
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET:
            del features["token_type_ids"]
        
        return features

    
    # TODO
    def split_into_chunks(self, examples, chunk_len: Optional[int] = None, stride: Optional[int] = None):
        pass 


    def _random_flip_question(self, example):
        words = example['question'].split()
        words[-1] = words[-1][:-1] + "," # removing question mark
        num_words = len(words)
        
        if num_words < 5:
            return example
        
        flip_idx = random.randint(int(0.2 * num_words), int(0.8 * num_words))
        words[flip_idx-1] = words[flip_idx-1] + "?" # adding question mark
        words = words[flip_idx:] + words[:flip_idx]
        return {'question': " ".join(words)}


    def flip_questions(self, dataset: Dataset = None, repeats: int = 1, concat: bool = True):
        if dataset is None:
            dataset = self.train_dataset

        new_dataset = dataset.map(lambda e: e)

        for i in range(repeats):
            fliped_dataset = dataset.map(self._random_flip_question, batched=False)
            new_dataset = concatenate_datasets([new_dataset, fliped_dataset])

        return new_dataset

    def prepare_train_features(self, examples):

        pad_on_right = self.tokenizer.padding_side == "right"

        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = self.get_tokenized_features(examples)
        if self.is_train :
            tokenized_examples = self.masking_input_ids(tokenized_examples)

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
        offset_mapping = tokenized_examples["offset_mapping"]

        # 데이터셋에 "start position", "enc position" label을 부여합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["example_id"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)  # cls index

            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            answers = examples[self.answer_column][sample_index]

            tokenized_examples["example_id"].append(examples["id"][sample_index])

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
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # text에서 current span의 End token index
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
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

    def prepare_test_features(self, examples):

        pad_on_right = self.tokenizer.padding_side == "right"

        tokenized_examples = self.get_tokenized_features(examples)
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples


    def get_datasets(self, include_test: bool = False):
        if include_test:
            self.train_datasets["test"] = self.test_datasets["validation"]
        return self.train_datasets

    def get_train_examples(self, data_dir: Optional[str] = None) -> Dataset:
        """Gets a new dataset of :class:`datasets.Dataset` for the train examples."""
        if data_dir is not None:
            self.train_dataset = self._load_dataset(data_dir, train=True)
            # self.train_dataset = self.train_dataset.map(self._flatten_multiple_answers, batched=True, batch_size=1, streaming=True)
        return self.train_dataset

    def get_eval_examples(self, data_dir: Optional[str] = None) -> Dataset:
        """Gets a new dataset of :class:`datasets.Dataset` for the eval examples."""
        if data_dir is not None:
            self.eval_dataset = self._load_dataset(data_dir, train=False)
            # self.eval_dataset = self.eval_dataset.map(self._flatten_multiple_answers, batched=True, batch_size=1, streaming=True)
        return self.eval_dataset

    def get_dev_examples(self, data_dir: Optional[str] = None) -> Dataset:
        """Gets a new dataset of :class:`datasets.Dataset` for the eval examples."""
        return self.get_eval_examples(data_dir=data_dir)

    def get_test_examples(self, data_dir: Optional[str] = None) -> Dataset:
        """Gets a new dataset of :class:`datasets.Dataset` for the test examples."""
        if data_dir is not None:
            self.eval_dataset = self._load_dataset(data_dir, train=False)
        return self.test_dataset


    def get_train_features(self, dataset: Dataset = None, set_format: bool = False, remove_columns: Optional[List[str]] = None) -> Dataset:
        """Gets a new dataset of class :class:`datasets.Dataset` for the train features."""
        if dataset is None:
            dataset = self.train_dataset

        if remove_columns is None:
            remove_columns = dataset.column_names

        self.is_train = True

        self.train_features = dataset.map(self.prepare_train_features, batched=True, batch_size=32, remove_columns=remove_columns, num_proc=4)

        if self.dataset_args.token_masking_with_normal_data:
            self.is_train = False
            self.non_mask_features = dataset.map(self.prepare_train_features, batched=True, batch_size=32, remove_columns=remove_columns, num_proc=4)
            assert self.train_features.features.type == self.non_mask_features.features.type
            self.train_features = concatenate_datasets([self.train_features, self.non_mask_features])
        
        if set_format:
            self.train_features.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "start_positions", "end_positions"])
        
        return self.train_features

    def get_eval_features(self, dataset: Dataset = None, set_format: bool = False, remove_columns: Optional[List[str]] = None) -> Dataset:
        """Gets a new dataset of class :class:`datasets.Dataset` for the eval features."""
        if dataset is None:
            dataset = self.eval_dataset

        if remove_columns is None:
            remove_columns = dataset.column_names

        self.is_train = False

        self.eval_features = dataset.map(self.prepare_train_features, batched=True, batch_size=32, remove_columns=remove_columns)
        
        if set_format:
            # remove_columns = [col for col in dataset.column_names if col not in ["input_ids", "token_type_ids", "attention_mask", "start_positions", "end_positions"]]
            # print(remove_columns)
            # self.eval_features = dataset.remove_columns(remove_columns)
            self.eval_features.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "start_positions", "end_positions"])

        return self.eval_features
    
    def get_dev_features(self, dataset: Dataset = None, set_format: bool = False, remove_columns: Optional[List[str]] = None) -> Dataset:
        """same method as `get_eval_features()`"""
        return self.get_eval_features(self, dataset=dataset, set_format=set_format, remove_columns=remove_columns)

    def get_test_features(self, dataset: Dataset = None, set_format: bool = False, remove_columns: Optional[List[str]] = None) -> Dataset:
        """Gets a new dataset of class :class:`datasets.Dataset` for the test features."""
        if dataset is None:
            dataset = self.test_dataset

        if remove_columns is None:
            remove_columns = dataset.column_names
        
        self.is_train = False

        self.test_features = dataset.map(self.prepare_test_features, batched=True, batch_size=32, remove_columns=remove_columns)

        if set_format:
            self.test_datasets.set_format(type="torch",  columns=["input_ids", "token_type_ids", "attention_mask"])

        return self.test_features

    def postprocess_qa_predictions(
        self,
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        is_world_process_zero: bool = True,
    ):
        """
        Post-processes : qa model의 prediction 값을 후처리하는 함수
        모델은 start logit과 end logit을 반환하기 때문에, 이를 기반으로 original text로 변경하는 후처리가 필요함

        Args:
            examples: 전처리 되지 않은 데이터셋 (see the main script for more information).
            features: 전처리가 진행된 데이터셋 (see the main script for more information).
            predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
                모델의 예측값 :start logits과 the end logits을 나타내는 two arrays              첫번째 차원은 :obj:`features`의 element와 갯수가 맞아야함.
            version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
                정답이 없는 데이터셋이 포함되어있는지 여부를 나타냄
            n_best_size (:obj:`int`, `optional`, defaults to 20):
                답변을 찾을 때 생성할 n-best prediction 총 개수
            max_answer_length (:obj:`int`, `optional`, defaults to 30):
                생성할 수 있는 답변의 최대 길이
            null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
                null 답변을 선택하는 데 사용되는 threshold
                : if the best answer has a score that is less than the score of
                the null answer minus this threshold, the null answer is selected for this example (note that the score of
                the null answer for an example giving several features is the minimum of the scores for the null answer on
                each feature: all features must be aligned on the fact they `want` to predict a null answer).
                Only useful when :obj:`version_2_with_negative` is :obj:`True`.
            output_dir (:obj:`str`, `optional`):
                아래의 값이 저장되는 경로
                dictionary : predictions, n_best predictions (with their scores and logits) if:obj:`version_2_with_negative=True`,
                dictionary : the scores differences between best and null answers
            prefix (:obj:`str`, `optional`):
                dictionary에 `prefix`가 포함되어 저장됨
            is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
                이 프로세스가 main process인지 여부(logging/save를 수행해야 하는지 여부를 결정하는 데 사용됨)
        """
        assert (
            len(predictions) == 2
        ), "`predictions` should be a tuple with two elements (start_logits, end_logits)."
        all_start_logits, all_end_logits = predictions

        assert len(predictions[0]) == len(
            features
        ), f"Got {len(predictions[0])} predictions and {len(features)} features."

        # example과 mapping되는 feature 생성
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # prediction, nbest에 해당하는 OrderedDict 생성합니다.
        all_predictions = OrderedDict()
        all_nbest_json = OrderedDict()
        if version_2_with_negative:
            scores_diff_json = OrderedDict()

        # Logging.
        logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
        logger.info(
            f"Post-processing {len(examples)} example predictions split into {len(features)} features."
        )

        # 전체 example들에 대한 main Loop
        for example_index, example in enumerate(tqdm(examples)):
            # 해당하는 현재 example index
            feature_indices = features_per_example[example_index]

            min_null_prediction = None
            prelim_predictions = []

            # 현재 example에 대한 모든 feature 생성합니다.
            for feature_index in feature_indices:
                # 각 featureure에 대한 모든 prediction을 가져옵니다.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # logit과 original context의 logit을 mapping합니다.
                offset_mapping = features[feature_index]["offset_mapping"]
                # Optional : `token_is_max_context`, 제공되는 경우 현재 기능에서 사용할 수 있는 max context가 없는 answer를 제거합니다
                token_is_max_context = features[feature_index].get(
                    "token_is_max_context", None
                )

                # minimum null prediction을 업데이트 합니다.
                feature_null_score = start_logits[0] + end_logits[0]
                if (
                    min_null_prediction is None
                    or min_null_prediction["score"] > feature_null_score
                ):
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                    }

                # `n_best_size`보다 큰 start and end logits을 살펴봅니다.
                start_indexes = np.argsort(start_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()

                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # out-of-scope answers는 고려하지 않습니다.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                        ):
                            continue
                        # 길이가 < 0 또는 > max_answer_length인 answer도 고려하지 않습니다.
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue
                        # 최대 context가 없는 answer도 고려하지 않습니다.
                        if (
                            token_is_max_context is not None
                            and not token_is_max_context.get(str(start_index), False)
                        ):
                            continue

                        ### TODO:
                        if (start_index == 0):
                            continue
                        
                        prelim_predictions.append(
                            {
                                "offsets": (
                                    offset_mapping[start_index][0],
                                    offset_mapping[end_index][1],
                                ),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )

            if version_2_with_negative:
                # minimum null prediction을 추가합니다.
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            # 가장 좋은 `n_best_size` predictions만 유지합니다.
            predictions = sorted(
                prelim_predictions, key=lambda x: x["score"], reverse=True
            )[:n_best_size]

            # 낮은 점수로 인해 제거된 경우 minimum null prediction을 다시 추가합니다.
            if version_2_with_negative and not any(
                p["offsets"] == (0, 0) for p in predictions
            ):
                predictions.append(min_null_prediction)

            # offset을 사용하여 original context에서 answer text를 수집합니다.
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0] : offsets[1]]

            # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
            if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""
            ):

                predictions.insert(
                    0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
                )

            # 모든 점수의 소프트맥스를 계산합니다(we do it with numpy to stay independent from torch/tf in this file, using the LogSumExp trick).
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # 예측값에 확률을 포함합니다.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # best prediction을 선택합니다.
            if not version_2_with_negative:
                all_predictions[example["id"]] = predictions[0]["text"]
            else:
                # else case : 먼저 비어 있지 않은 최상의 예측을 찾아야 합니다
                i = 0
                while predictions[i]["text"] == "":
                    i += 1
                best_non_null_pred = predictions[i]

                # threshold를 사용해서 null prediction을 비교합니다.
                score_diff = (
                    null_score
                    - best_non_null_pred["start_logit"]
                    - best_non_null_pred["end_logit"]
                )
                scores_diff_json[example["id"]] = float(score_diff)  # JSON-serializable 가능
                if score_diff > null_score_diff_threshold:
                    all_predictions[example["id"]] = ""
                else:
                    all_predictions[example["id"]] = best_non_null_pred["text"]

            # np.float를 다시 float로 casting -> `predictions`은 JSON-serializable 가능
            all_nbest_json[example["id"]] = [
                {
                    k: (
                        float(v)
                        if isinstance(v, (np.float16, np.float32, np.float64))
                        else v
                    )
                    for k, v in pred.items()
                }
                for pred in predictions
            ]

        # output_dir이 있으면 모든 dicts를 저장합니다.
        if output_dir is not None:
            assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

            prediction_file = os.path.join(
                output_dir,
                "predictions.json" if prefix is None else f"predictions_{prefix}".json,
            )
            nbest_file = os.path.join(
                output_dir,
                "nbest_predictions.json"
                if prefix is None
                else f"nbest_predictions_{prefix}".json,
            )
            if version_2_with_negative:
                null_odds_file = os.path.join(
                    output_dir,
                    "null_odds.json" if prefix is None else f"null_odds_{prefix}".json,
                )

            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
                )
            logger.info(f"Saving nbest_preds to {nbest_file}.")
            with open(nbest_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n"
                )
            if version_2_with_negative:
                logger.info(f"Saving null_odds to {null_odds_file}.")
                with open(null_odds_file, "w", encoding="utf-8") as writer:
                    writer.write(
                        json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + "\n"
                    )

        return all_predictions

    def post_processing_function(
        self,
        examples, 
        features, 
        predictions, 
        training_args
    ):
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = self.postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=self.dataset_args.max_ans_len,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[self.answer_column]}
                for ex in self.eval_dataset
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    def visualize_plots(self, remove_columns: Optional[List[str]] = None):

        import matplotlib.pyplot as plt

        self.numeric_columns = []
        self.categorical_columns = []
        self.string_columns = []

        for c in self.train_dataset.column_names:
            if remove_columns is not None and c in remove_columns:
                continue

            if isinstance(self.train_dataset[c][0], int):

                if len(self.train_dataset.unique(c)) < 10:
                    self.categorical_columns.append(c)
                else:
                    self.numeric_columns.append(c)
                
                plt.hist(self.train_dataset[c])
                plt.title("Histogram of " + c)
                plt.show()

            elif isinstance(self.train_dataset[c][0], float):

                if len(self.train_dataset.unique(c)) < 10:
                    self.categorical_columns.append(c)
                else:
                    self.numeric_columns.append(c)
                
                plt.hist(self.train_dataset[c])
                plt.title("Histogram of " + c)
                plt.show()

            elif isinstance(self.train_dataset[c][0], bool):
                self.categorical_columns.append(c)
                
            elif isinstance(self.train_dataset[c][0], str):
                
                self.string_columns.append(c)
                self.numeric_columns.append("len_" + c)
                if len(self.train_dataset.unique(c)) < 10:
                    self.categorical_columns.append(c)

                plt.hist(self.train_dataset.map(
                    lambda e: {"length": len(e[c])})["length"])
                plt.title("Length of " + c)
                plt.show()


    def correlation_plots(self):

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(self.numeric_columns), len(self.numeric_columns), figsize=(10, 10))
        mybox={'facecolor':'y','edgecolor':'r','boxstyle':'round','alpha':0.5}

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i in range(len(self.numeric_columns)):
            for j in range(len(self.numeric_columns)):
                i_name = self.numeric_columns[i]
                j_name = self.numeric_columns[j]

                if i_name.count("len") == 0:
                    i_data = self.train_dataset[i_name]
                else:
                    i_name = i_name[4:]
                    i_data = self.train_dataset.map(lambda e: {"length": len(e[i_name])})["length"]

                if j_name.count("len") == 0:
                    j_data = self.train_dataset[j_name]
                else:
                    j_name = j_name[4:]
                    j_data = self.train_dataset.map(lambda e: {"length": len(e[j_name])})["length"]
                
                if i == j:
                    axes[i][j].hist(i_data, color=colors[j])
                    axes[i][j].text(x=0, y=200, s=i_name, bbox=mybox,)

                else:
                    axes[i][j].scatter(
                        i_data, 
                        j_data, 
                        s=0.25, alpha=0.2, c=colors[j]
                    )

        plt.show()

    def masking_input_ids(self, examples):
        
        CLS_TOKEN = self.tokenizer.cls_token_id
        SEP_TOKEN = self.tokenizer.sep_token_id
        MASK_TOKEN = self.tokenizer.mask_token_id
        MASK_RATIO = self.dataset_args.token_masking_ratio
        MAX_MASK_NUM = self.dataset_args.token_masking_max

        new_input_ids = []
        past_question = []
        past_masked_question = []
        for question_include_context_ids in examples['input_ids']:
            
            question = []
            for input_id in question_include_context_ids:
                if input_id == CLS_TOKEN :
                    continue
                if input_id == SEP_TOKEN :
                    break
                question.append(input_id)
            
            new_sentence = past_question != question

            past_question = question

            if new_sentence:
                mask = np.random.rand(len(question)) < MASK_RATIO
                if sum(mask) > MAX_MASK_NUM:
                    mask_idx = np.where(mask)
                    set_false_pos = np.random.choice(mask_idx[0], sum(mask) - MAX_MASK_NUM, replace=False)
                    mask[set_false_pos] = False
                masked_question = [MASK_TOKEN if m else word for word, m in zip(question, mask)]
            else :
                masked_question = past_masked_question
            
            question_masked_ids = [CLS_TOKEN] + masked_question + [SEP_TOKEN] + question_include_context_ids[len(question)+2:]
            past_masked_question = masked_question
            new_input_ids.append(question_masked_ids)
        
        examples['input_ids'] = new_input_ids
        return examples