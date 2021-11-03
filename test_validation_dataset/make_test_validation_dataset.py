# cd ~
# python ./test_validation_dataset/make_test_validation_dataset.py

import pandas as pd
from datasets import Dataset

def main():
    test_df = pd.read_csv('./test_validation_dataset/test_merge_data.csv')
    test_df = test_df[['id', 'question' , 'context', 'answer_text', 'answer_start', 'title']]
    test_df['document_id'] = [i for i in range(len(test_df))]
    answers = []
    for i, row in test_df.iterrows():
        answers.append({'answer_start': [row['answer_start']], 'text': [row['answer_text']]})
    test_df['answers'] = answers
    test_df.drop(columns=['answer_text', 'answer_start'], inplace=True)

    test_validation_dataset = Dataset.from_pandas(test_df)
    test_validation_dataset.save_to_disk('/opt/ml/data/test_validation_dataset')

if __name__ == '__main__':
    main()