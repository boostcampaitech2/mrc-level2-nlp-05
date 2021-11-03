import pandas as pd
import json

# Aistages output.csv 다운로드, json으로 불러와야 한다. (1위부터 10위까지)
# 처음 파일명인 output (숫자).csv를 output숫자.csv 로 변경해줘야함 !!
def get_ready_output(predictions_num: int) :
    outputs = pd.DataFrame()
    for i in range(1, predictions_num + 1) : # 1부터 json 파일의 개수 + 1
        with open(f'/opt/ml/mrc-level2-nlp-05/ipynb/ensemble/output{i}.json', 'r') as f:
            output = json.load(f)
        outputs = pd.concat([outputs, pd.DataFrame({'prediction': output})], axis=1)
        
    # id 값을 얻기위해 transpose
    outputs_t = outputs.T

    return outputs, outputs_t


# 10개의 결과를 다운로드 받았다면 10
outputs, outputs_t = get_ready_output(10)

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
    final_predictions.to_json('final_predictions.json', force_ascii=False, index=True)

    return final_predictions


get_final_predictions(outputs, outputs_t)

# 다시 불러오기 test
with open(f'/opt/ml/mrc-level2-nlp-05/ipynb/ensemble/final_predictions.json', 'r') as f:
    output = json.load(f)
outputs = pd.DataFrame({'prediction': output})