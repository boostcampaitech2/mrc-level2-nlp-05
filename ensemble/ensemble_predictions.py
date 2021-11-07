import pandas as pd
import json
import os

def get_ready_outputs(folder, paths) :
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


def get_final_predictions(outputs, outputs_t, save_file):
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
    final_predictions.to_json(save_file, force_ascii=False, index=True, indent=4)

    return final_predictions


def get_final_choose_shorter(outputs, outputs_t, save_file):
    final_predictions = pd.DataFrame()
    output = []
    index = []
    # 10개의 prediction 중 가장 많은 예측된 결과를 output에 저장, index는 index에 저장
    for idx, row in outputs.iterrows():
        preds = outputs_t[idx].value_counts()
        if len(preds) >= 2 & preds[0] == preds[1] : # 두 개의 답이 동일하게 득표 -> 길이가 짧은 답 선택
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
    final_predictions.to_json(save_file, force_ascii=False, index=True, indent=4)

    return final_predictions


def get_final_choose_longer(outputs, outputs_t, save_file):
    final_predictions = pd.DataFrame()
    output = []
    index = []
    # 10개의 prediction 중 가장 많은 예측된 결과를 output에 저장, index는 index에 저장
    for idx, row in outputs.iterrows():
        preds = outputs_t[idx].value_counts()
        if len(preds) >= 2 & preds[0] == preds[1] :# 두 개의 답이 동일하게 득표 -> 길이가 긴 답 선택
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
    final_predictions.to_json(save_file, force_ascii=False, index=True, indent=4)

    return final_predictions

def main():
    # predictions 파일들을 넣어둔 폴더명을 folder 변수에 넣으면 그 폴더 안의 모든 파일을 읽어온다.
    pred_folder = os.path.join('ensemble', 'source')
    pred_file_list = os.listdir(f'./{pred_folder}')
    save_folder = os.path.join('ensemble', 'result')
    save_file = os.path.join(save_folder, 'exp000_predictions.json')

    outputs, outputs_t = get_ready_outputs(pred_folder, pred_file_list)

    get_final_predictions(outputs, outputs_t, save_file)
    #get_final_choose_shorter(outputs, outputs_t, save_file_name)
    #get_final_choose_longer(outputs, outputs_t, save_file_name)

if __name__ == '__main__':
    main()