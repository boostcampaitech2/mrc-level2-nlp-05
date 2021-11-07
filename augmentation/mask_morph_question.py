from transformers import AutoTokenizer
import numpy as np
from datasets import load_from_disk
from pororo import Pororo
import logging


logger = logging.getLogger(__name__)

model_name = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

MASK_TOKEN = tokenizer.mask_token
MASK_RATIO = 0.2
MAX_MASK_NUM = 2
LOAD_DIR = "/opt/ml/data/train_dataset"
SAVE_DIR_WORD = "/opt/ml/data/mask_morph_q_train_dataset"


ner = Pororo(task="ner", lang="ko")

def mask_ner_tag(examples):
    questions = examples["question"]
    aug_questions = []
    q_mark = '?'
    for question in questions:
        pairs = ner(question)
        mask = [False if tag == 'O' else True for _, tag in pairs]
        mask = np.array(mask)
        
        # set maximum num of masks to MAX_MASK_NUM
        if sum(mask) > MAX_MASK_NUM:
            mask_idx = np.where(mask)
            set_false_pos = np.random.choice(mask_idx[0], sum(mask) - MAX_MASK_NUM, replace=False)
            mask[set_false_pos] = False
        
        masked_text = [MASK_TOKEN if mask else pair[0] for pair, mask in zip(pairs, mask)]
        if mask[-1]: masked_text.append(q_mark)   # if the last word is masked, append '?'
        aug_questions.append("".join(masked_text))

    return {"question": aug_questions}


# load & map augmentation
datasets = load_from_disk("/opt/ml/data/train_dataset")
train_dataset = datasets["train"]

# train_dataset = train_dataset.select(range(30))  # sample
train_dataset_aug = train_dataset.map(
                        mask_ner_tag,
                        batched=True,
                        batch_size=8,
                        # num_proc=4
                    )

# save datasets
train_dataset_aug.save_to_disk(SAVE_DIR_WORD)

logger.info(f'Created a new train dataset of size {len(train_dataset_aug)}')
