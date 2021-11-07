from pororo import Pororo
from datasets import load_from_disk
import numpy as np
from typing import List, Tuple
import logging


logger = logging.getLogger(__name__)

ner = Pororo(task="ner", lang="ko")

ENG_TO_KOR = {'PERSON' : '사람', 'LOCATION' : '장소', 'ORGANIZATION' : '조직', 'ARTIFACT' : '인공물',
            'DATE' : '날짜', 'TIME' : '시간', 'CIVILIZATION' : '역할', 'ANIMAL' : '동물',
            'PLANT' : '식물', 'QUANTITY' : '수량', 'STUDY_FIELD' : '분야', 'THEORY' : '이론',
            'EVENT' : '사건', 'MATERIAL' : '물질', 'TERM' : '용어', 
            'OCCUPATION' : '직업', 'COUNTRY' : '국가', 'CITY' : '도시', 'DISEASE' :'질병'}  # 전체 태그

IGNORE_TAGS = ('O', 'ARTIFACT', 'DATE', 'TIME', 'CIVILIZATION', 'QUANTITY', 'TERM')  # 부자연스러운 태그


def random_aug(
    word_tag_pairs: List[Tuple[str, str]],
    valid_tag: List[bool],
    max_tag_num: int = 2) -> str:
    """
    문장에 NER 태그를 random하게 추가해주는 함수.

    Args:
        word_tag_pairs: 입력 문장의 단어와 NER 태그 쌍을 담은 리스트
        valid_tag: 의미 있는 태그를 갖는지에 대한 여부를 나타내는 리스트
        max_tag_num: 출력 문장에 추가할 태그의 최대 개수
    Returns:
        aug_sentence: NER 태그가 단어 앞에 추가된 문장
    """

    tag_num = sum(valid_tag)  # total num of tags
    
    if tag_num > max_tag_num:
        valid_tag = np.array(valid_tag)
        tag_pos_idx = np.where(valid_tag)

        # leave only max_tag_num positions to True
        set_false_pos = np.random.choice(tag_pos_idx[0], tag_num - max_tag_num, replace=False)
        valid_tag[set_false_pos] = False
    
    aug_sentence = ''
    # create a new sentence with tags inserted
    for item, add_tag in zip(word_tag_pairs, valid_tag):
        if add_tag and item[1] not in IGNORE_TAGS:
            aug_sentence += f'{ENG_TO_KOR[item[1]]} '
        aug_sentence += item[0]      
    return aug_sentence


def add_ner_tag(examples):
    """
    데이터 증강 시 map에서 적용할 함수.
    batch 데이터를 받으면 "question" 문장들을 변형시켜 batch 데이터를 반환함.
    """
    new_questions = []
    for question in examples["question"]:
        ner_question = ner(question)  # list of (word, tag)
        tag_pos = [False if word[1] in IGNORE_TAGS else True for word in ner_question]
        new_questions.append(random_aug(ner_question, tag_pos, max_tag_num=2))
    return {"question": new_questions}


# load datasets
datasets = load_from_disk("/opt/ml/data/train_dataset")
train_dataset = datasets["train"]


# data augmentation
train_dataset_aug = train_dataset.map(
                        add_ner_tag,
                        batched=True,
                        batch_size=8,
                        # num_proc=2  # Pororo NER에서 에러 발생
                    )

datasets["train"] = train_dataset_aug
datasets.save_to_disk("/opt/ml/data/ner_only_train_dataset")

logger.info(f'Created a new train dataset of size {len(train_dataset_aug)}')
