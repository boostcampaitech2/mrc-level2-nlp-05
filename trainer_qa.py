# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Question-Answering task와 관련된 'Trainer'의 subclass 코드 입니다.
"""

import time
import math
from typing import Callable, List

import datasets
from datasets import Dataset

from transformers.trainer import Trainer
from transformers.trainer_utils import speed_metrics

# Huggingface의 Trainer를 상속받아 QuestionAnswering을 위한 Trainer를 생성합니다.
class QATrainer(Trainer):
    def __init__(
        self, 
        *args, 
        eval_examples: Dataset=None, 
        post_process_function: Callable=None, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(
        self, 
        eval_dataset: Dataset=None, 
        eval_examples: Dataset=None, 
        ignore_keys: List[str]=None,
    ):
        self._memory_tracker.start()

        eval_dataset  = self.eval_dataset  if eval_dataset is None  else eval_dataset
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        temp_compute_metrics = self.compute_metrics
        self.compute_metrics = None

        try:
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if temp_compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix="eval"
            )
        finally:
            self.compute_metrics = temp_compute_metrics
        
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys())
            )

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                examples=eval_examples, 
                features=eval_dataset, 
                predictions=output.predictions, 
                training_args=self.args, 
            )
            
            output.metrics.update({"eval_" + k: v for k, v in self.compute_metrics(eval_preds).items()})
        
        total_batch_size = self.args.eval_batch_size * self.args.world_size

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self.log(output.metrics)
        self.log(
            speed_metrics(
                "eval",
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        return output.metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"],
                columns=list(test_dataset.features.keys()),
            )

        predictions = self.post_process_function(
            test_examples, test_dataset, output.predictions, self.args
        )
        return predictions
