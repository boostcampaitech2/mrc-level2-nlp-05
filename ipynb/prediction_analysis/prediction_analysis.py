import pandas as pd
from datasets import load_from_disk


class PredictionAnalysis:

    data_path = "/opt/ml/data/train_dataset/"

    def __init__(self, file_path):
        self.file_path = file_path
        self.make_prediction_df()
        self.make_target_df()
        self.merge_df()

    def make_prediction_df(self):
        '''
        make prediction's dataframe
        '''
        self.df_pred = pd.read_json(self.file_path, orient="index")
        self.df_pred = self.df_pred.reset_index()
        self.df_pred.columns = ["id", "prediction"]

    def make_target_df(self):
        '''
        make target's dataframe
        '''
        dataset = load_from_disk(self.data_path)
        self.df_target = pd.DataFrame(dataset["validation"])

        # "answers" column 확장
        answer_start_list = []
        answer_text_list = []

        for _, row in self.df_target.iterrows():
            answer_start_list.append(row.answers['answer_start'][0])
            answer_text_list.append(row.answers['text'][0])

        self.df_target['answer_start'] = answer_start_list
        self.df_target['answer_text'] = answer_text_list
    
    def merge_df(self):
        '''
        merge prediction and target dataframes
        '''
        self.df_merged = pd.merge(self.df_pred, self.df_target, how='inner', on='id')
        self.df_merged = self.df_merged[['id', 'title', 'context', 'question', 'document_id', 'answer_start', 'answer_text', 'prediction']]
        self.df_merged["correct"] = self.df_merged["answer_text"] == self.df_merged["prediction"]
