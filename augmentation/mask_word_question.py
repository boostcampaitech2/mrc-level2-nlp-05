from transformers import AutoTokenizer
from datasets import load_from_disk, concatenate_datasets
import numpy as np
from pororo import Pororo
from tqdm import tqdm
import logging



logger = logging.getLogger(__name__)

model_name = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

MASK_RATIO = 0.2
MAX_MASK_NUM = 2
MASK_TOKEN = tokenizer.mask_token  # '[MASK]'

LOAD_DIR = "/opt/ml/data/train_dataset"
SAVE_DIR_WORD = "/opt/ml/data/mask_word_q_train_dataset"

def mask_word(examples):
    questions = examples["question"]
    aug_questions = []
    q_mark = '?'
    for question in questions:
        words = question.split()
        
        mask = np.random.rand(len(words)) < MASK_RATIO
        
        # set maximum num of masks to MAX_MASK_NUM
        if sum(mask) > MAX_MASK_NUM:
            mask_idx = np.where(mask)
            set_false_pos = np.random.choice(mask_idx[0], sum(mask) - MAX_MASK_NUM, replace=False)
            mask[set_false_pos] = False

        masked_text = [MASK_TOKEN if mask else word for word, mask in zip(words, mask)]
        if mask[-1]: masked_text.append(q_mark)   # if the last word is masked, append '?'
        aug_questions.append(" ".join(masked_text))

    return {"question": aug_questions}


# load & map augmentation
datasets = load_from_disk(LOAD_DIR)
train_dataset = datasets["train"]

# train_dataset = train_dataset.select(range(30))  # sample
train_dataset_aug = train_dataset.map(
                        mask_word,
                        batched=True,
                        batch_size=8,
                        num_proc=4
                    )

# save datasets
train_dataset_aug.save_to_disk(SAVE_DIR_WORD)

logger.info(f'Created a new train dataset of size {len(train_dataset_aug)}')
