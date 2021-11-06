# import sys
# sys.path.append('..')

from arguments import DatasetArguments
from processor import QAProcessor

import re
import numpy as np

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoConfig, AutoModel
from sentence_transformers import SentenceTransformer, util

def main():
    model_name = 'klue/roberta-large'

    print(f"Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading sentence transformers model")
    st_model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

    def get_embeddings(sentence):
        tokenized_sentence = tokenizer(
            sentence,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'    
        )

        input_ids = tokenized_sentence['input_ids']
        batch_size = input_ids.shape[0]

        embeddings = []
        for i in range(batch_size):
            decoded_tokens = tokenizer.decode(input_ids[i])
            embedded_tokens = st_model.encode(decoded_tokens)
            embeddings.append(embedded_tokens)
        
        return np.array(embeddings)

    def make_mask_context(example):
        k = 2
        mask_token_id = tokenizer.mask_token_id
        sep_token_id = tokenizer.sep_token_id
        pad_token_id = tokenizer.pad_token_id

        input_id = example['input_ids']

        # sep 토큰 및 pad 토큰 index 획득
        sep_idx = input_id.index(sep_token_id)
        pad_idx = -1
        if pad_token_id in input_id:
            pad_idx = input_id.index(pad_token_id)
        
        # 질문 및 정답 문장 추출
        question = tokenizer.decode(input_id[1:sep_idx])
        answer = tokenizer.decode(input_id[example['start_positions']:example['end_positions']+1])

        # context에 해당하는 input_id 추출
        context_ids = input_id[sep_idx+2:-1] if pad_idx > -1 else input_id[sep_idx+2:pad_idx-1]

        # 질문 문장 임베딩
        question_embedding = get_embeddings(question)

        # context의 각 토큰 임베딩
        tokens = tokenizer.convert_ids_to_tokens(context_ids)
        context_token_embeddings = get_embeddings(tokens)

        # 질문과 context의 각 토큰 사이의 유사도 계산
        cosine_scores = util.pytorch_cos_sim(question_embedding, context_token_embeddings).squeeze(0)

        # 마스킹 대상 토큰 필터링
        results = [(idx+sep_idx+2, token, sim.item()) for idx, (token, sim) in enumerate(zip(tokens, cosine_scores))]
        results = [(idx, token, sim) for idx, token, sim in results if '##' not in token]
        results = [(idx, token, sim) for idx, token, sim in results if re.match(r"[가-힣]+", token) is not None]
        results = sorted(results, key=lambda x: x[2], reverse=True)

        # 유사도 상위 k개의 토큰
        best_sim_tokens = []
        for _, token, _ in results:
            if len(best_sim_tokens) == k:
                break
            if token not in best_sim_tokens:
                best_sim_tokens.append(token)
        
        # 마스킹 대상 토큰
        mask_tokens = []
        for idx, token, _ in results:
            if token in best_sim_tokens:
                mask_tokens.append((idx, token))

        # 마스크 토큰으로 치환
        for idx, _ in mask_tokens:
            input_id[idx] = mask_token_id        

        example['input_ids'] = input_id

        return example

    def make_mask_context_batch(examples):
        k = 2
        mask_token_id = tokenizer.mask_token_id
        sep_token_id = tokenizer.sep_token_id
        pad_token_id = tokenizer.pad_token_id

        for i in range(len(examples['input_ids'])):

            input_id = examples['input_ids'][i]

            # sep 토큰 및 pad 토큰 index 획득
            sep_idx = input_id.index(sep_token_id)
            pad_idx = -1
            if pad_token_id in input_id:
                pad_idx = input_id.index(pad_token_id)
            
            # 질문 및 정답 문장 추출
            question = tokenizer.decode(input_id[1:sep_idx])
            answer = tokenizer.decode(input_id[examples['start_positions'][i]:examples['end_positions'][i]+1])

            # context에 해당하는 input_id 추출
            context_ids = input_id[sep_idx+2:-1] if pad_idx > -1 else input_id[sep_idx+2:pad_idx-1]

            # 질문 문장 임베딩
            question_embedding = get_embeddings(question)

            # context의 각 토큰 임베딩
            tokens = tokenizer.convert_ids_to_tokens(context_ids)
            context_token_embeddings = get_embeddings(tokens)

            # 질문과 context의 각 토큰 사이의 유사도 계산
            cosine_scores = util.pytorch_cos_sim(question_embedding, context_token_embeddings).squeeze(0)

            # 마스킹 대상 토큰 필터링
            results = [(idx+sep_idx+2, token, sim.item()) for idx, (token, sim) in enumerate(zip(tokens, cosine_scores))]
            results = [(idx, token, sim) for idx, token, sim in results if '##' not in token]
            results = [(idx, token, sim) for idx, token, sim in results if re.match(r"[가-힣]+", token) is not None]
            results = sorted(results, key=lambda x: x[2], reverse=True)

            # 유사도 상위 k개의 토큰
            best_sim_tokens = []
            for _, token, _ in results:
                if len(best_sim_tokens) == k:
                    break
                if token not in best_sim_tokens:
                    best_sim_tokens.append(token)
            
            #print(best_sim_tokens)

            # 마스킹 대상 토큰
            mask_tokens = []
            for idx, token, _ in results:
                if token in best_sim_tokens:
                    mask_tokens.append((idx, token))

            # 마스크 토큰으로 치환
            for idx, _ in mask_tokens:
                input_id[idx] = mask_token_id        

            examples['input_ids'][i] = input_id

        return examples

    dataset_args = DatasetArguments()

    processor = QAProcessor(dataset_args, tokenizer, concat=False)

    print(f"Loading dataset")
    mask_dataset = processor.get_train_features()

    print(f"Context Masking...")
    mask_dataset = mask_dataset.map(
        make_mask_context
    )
    # mask_dataset = mask_dataset.map(
    #     make_mask_context_batch,
    #     batched=True
    # )


    mask_dataset.save_to_disk('/opt/ml/data/context_mask_dataset')

if __name__ == '__main__':
    main()