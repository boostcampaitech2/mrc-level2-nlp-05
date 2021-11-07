import pandas as pd
import re
from datasets import load_from_disk


datasets = load_from_disk("/opt/ml/data/train_dataset")

train_df = pd.DataFrame(datasets["train"])
val_df = pd.DataFrame(datasets["validation"])

# == Answers ====================================================================================

def extract_answer_specials(df:pd.DataFrame):
    answer_specials = []
    for i, row in df.iterrows():
        text = row.answers["text"][0]
        result = re.findall(r"[^ㄱ-ㅎ가-힣A-Za-z\d\s]+", text)
        if len(result) > 0:
            answer_specials.extend(result)
    answer_specials = list(set(answer_specials))
    return answer_specials

train_answer_sepcials = extract_answer_specials(train_df)  # train set의 answers에 포함된 특수문자
val_answer_sepcials = extract_answer_specials(val_df)  # validation set의 answers에 포함된 특수문자


# == Context ====================================================================================
def extract_context_specials(df:pd.DataFrame):
    context_specials = []
    for text in df["context"]:
        # 중국어: 一-龥 / 일본어: ぁ-ゔァ-ヴー々〆〤 / 러시아어(적용 X): \u0400-\u04FF
        result = re.findall(r"[^ㄱ-ㅎ가-힣A-Za-z\d\s一-龥ぁ-ゔァ-ヴー々〆〤]+", text)
        if len(result) > 0:
            context_specials.extend(result)
    context_specials = list(set(context_specials))
    return context_specials

train_context_sepcials = extract_context_specials(train_df)  # train set의 context에 포함된 특수문자
val_context_sepcials = extract_context_specials(val_df)  # validation set의 context에 포함된 특수문자


print("train set의 answers에 포함된 특수문자를 제외한 context에 포함된 특수문자:")
train_specials = sorted([special for special in train_context_sepcials if special not in train_answer_sepcials])
print(train_specials)
print()

print("validation set의 answers에 포함된 특수문자를 제외한 context에 포함된 특수문자:")
val_specials = sorted([special for special in val_context_sepcials if special not in val_answer_sepcials])
print(val_specials)
