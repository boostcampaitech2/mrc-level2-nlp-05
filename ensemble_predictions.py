import pandas as pd
from typing import List
import json
import os

# output.csv 를 넣어둔 폴더명을 folder에 넣으면 그 폴더 안의 모든 파일을 읽어옴
folder = 'predictions'
paths = os.listdir(f'./{folder}')

# output.csv을 json으로 불러와야 한다.
def get_ready_outputs(paths:List) :
    outputs = pd.DataFrame()
    for path in paths :
        with open(f'./{folder}/{path}', 'r') as f:
            output = json.load(f)
        keys = []
        values = []
        for key, value in output.items():
            keys.append(key)
            values.append(value)
        df = pd.DataFrame({'keys':keys,'values':values})
        df = df.set_index('keys')
        outputs = pd.concat([outputs, df], axis=1, sort=False)
        outputs_t = outputs.T
    return outputs, outputs_t


outputs, outputs_t = get_ready_outputs(paths)


def get_final_predictions(outputs, outputs_t) :
    final_predictions = pd.DataFrame()
    output = []
    index = []
    # 10개의 prediction 중 가장 많은 예측된 결과를 output에 저장, index는 index에 저장
    for idx,row in outputs.iterrows():
        output.append(outputs_t[idx].value_counts().idxmax())
        index.append(idx)
    final_predictions = pd.concat([final_predictions, pd.Series(output)]) # series로 변환하여 추가
    # column명 변경, index를 id로 변경 
    final_predictions.columns = ['prediction']
    final_predictions.index = index
    # final_predictions를 series로 변환하여 json 파일로 내보내기
    final_predictions = final_predictions['prediction']
    final_predictions.to_json('final_predictions.json', force_ascii=False, index=True, indent=4)

    return final_predictions


def get_final_choose_shorter(outputs, outputs_t) :
    final_predictions = pd.DataFrame()
    output = []
    index = []
    # 10개의 prediction 중 가장 많은 예측된 결과를 output에 저장, index는 index에 저장
    for idx, row in outputs.iterrows():
        preds = outputs_t[idx].value_counts()
        if len(preds) >= 2 & preds[0] == preds[1] :
            output.append(preds.keys()[0]) if len(preds.keys()[0]) <= len(preds.keys()[1]) else output.append(preds.keys()[1])
        else :
            output.append(outputs_t[idx].value_counts().idxmax())
        index.append(idx)
    final_predictions = pd.concat([final_predictions, pd.Series(output)]) # series로 변환하여 추가
    # column명 변경, index를 id로 변경 
    final_predictions.columns = ['prediction']
    final_predictions.index = index
    # final_predictions를 series로 변환하여 json 파일로 내보내기
    final_predictions = final_predictions['prediction']
    final_predictions.to_json('final_predictions.json', force_ascii=False, index=True, indent=4)

    return final_predictions


def get_final_choose_longer(outputs, outputs_t) :
    final_predictions = pd.DataFrame()
    output = []
    index = []
    # 10개의 prediction 중 가장 많은 예측된 결과를 output에 저장, index는 index에 저장
    for idx, row in outputs.iterrows():
        preds = outputs_t[idx].value_counts()
        if len(preds) >= 2 & preds[0] == preds[1] :
            output.append(preds.keys()[0]) if len(preds.keys()[0]) >= len(preds.keys()[1]) else output.append(preds.keys()[1])
        else :
            output.append(outputs_t[idx].value_counts().idxmax())
        index.append(idx)
    final_predictions = pd.concat([final_predictions, pd.Series(output)]) # series로 변환하여 추가
    # column명 변경, index를 id로 변경 
    final_predictions.columns = ['prediction']
    final_predictions.index = index
    # final_predictions를 series로 변환하여 json 파일로 내보내기
    final_predictions = final_predictions['prediction']
    final_predictions.to_json('final_predictions.json', force_ascii=False, index=True, indent=4)

    return final_predictions


get_final_predictions(outputs, outputs_t)


# 다시 불러오기 test
with open(f'/opt/ml/mrc-level2-nlp-05/ipynb/ensemble/final_predictions.json', 'r') as f:
    output = json.load(f)
outputs = pd.DataFrame({'prediction': output})
