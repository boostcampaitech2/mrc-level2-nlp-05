# ODQA MRC - ㅇㄱㄹㅇ

## Boostcamp AI-Tech 2기

### DPR 진행 정도
- 베이스라인 작업 완료되면 한번 다시 작동 여부 확인 필요
- Trainer 쪽 eval loss 쪽 추가 얼른... -> WandB 연동으로 학습 확인 필요!

<br>

### 실험 및 기능 추가 예정
- 현재 multilingual BERT 기준, KLUE 베이스라인 기준으로도 성능 확인 후 parser 로 변경 가능하게 수정
- Tokenizer max_length -> 최대로 가져가기...! -> 실험
- `Retrieval.py`의 `main()` 으로 sparse, dense PR train set 에 대해 각각 출력하게 수정 *Accuracy + Precision @ k* 함수 추가하기


<br>

## Updates

* (1:10, Oct 21) 베이스라인 리팩토링 한 것을 올려놨습니다! 다 수정하지는 않았고, 일단 돌아가게만 해놓은 상태라고 보시면 됩니다. 자세한 내용 및 앞으로 구현해야 되는 사항은 아래와 같습니다.

  1. 🟢 Preprocessor 분리 - preprocessor.py로 데이터셋 전처리르 옮겨놨습니다. 또한, 중복되는 코드 역시도 refactoring 하였습니다. 추가적인 기능을 원하시는 분은 지난 대회와 마찬가지로 `Preprocessor`를 상속하는 클래스를 만들어주면 됩니다. 이후, `prepare_train_features()`, `prepare_eval_features()` 두 메서드를 구현해주면 됩니다. 저번 대회보다 개선된 사항은 `__init__` 시에 `dataset_args, tokenizer, column_names`를 받도록 설계하여, `set_tokenizer()`, `set_column_names()`를 별도로 실행하지 않아도 되게끔 했습니다.

  2. 🟡 Retriever Argument 생성 - 다양한 retriever를 사용할 수 있도록 일단 넣어놨습니다. 기본적인 retriever 구현은 이번주 일요일까지 수행 후 합치는 작업을 진행하겠습니다. (due Oct 24)

  3. 🟢 `increment_path()` 함수 - 잡다한 유틸리티는 `utils.py`에 모아놨습니다. 이름이 겹치는 경우에는 자동으로 이름에 suffix를 붙여줍니다.

  4. 🔴 custom model 구현 - argparser를 이용하여 huggingface 모델과 custom model을 동일한 인터페이스로 불러오는 작업을 수행해야 합니다. 아마 `get_model_and_tokenizer()` 형태로 똑같이 구현할 것 같습니다. (due Oct 22)

  5. 🔴 `post_processing_function`을 post_processing_class로 변경 - 현재는 함수로 구현되어 있지만, 이를 class로 구현해서 `__call__()` 메서드를 통해서 후처리를 수행해야 할 것으로 보입니다. 해당 부분이 지저분해져서 datasets를 넘기도록 코드가 구성되어 있기 때문에, 현재 `QuestionAnsweringTrainer` 클래스 구현까지도 수정된 상태입니다. (datasets, dataset_args 등의 인자를 임시로 추가하여 동작은 하도록 만들었습니다...) Preprocessor와 마찬가지로 사전에 설정해두어야 할 듯 하고, 나중에 ensemble이나 generation task로 문제를 해결할 경우 이에 적합한 post_processor를 만드는 것도 중요한 태스크일 것입니다. (due Oct 22)

  6. 🔴 inference.py 제작 - train.py와 매우 유사하기 때문에 재사용이 가능한 함수들은 refactoring하여 inference.py를 간소화해서 제작하는 것이 필요해 보입니다. (due Oct 24)

  7. 🔴 argparser 구현 - model, config, tokenizer, preprocessor, retriever 부분에 해당하는 argument parser 구현 및 연동을 작성해야 합니다. (due Oct 21)

  8. 🟡 custom classification head 구현 - custom model의 경우 다양한 백본 모델의 config를 토대로 자동으로 custom classification head를 붙여주는 코드를 제작할 예정입니다. 4번 항목과 연관된 내용입니다. (due Oct 22) 
