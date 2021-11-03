import os
import json
import time
import faiss
import pickle
import numpy as np
import pandas as pd
import torch

from rank_bm25 import BM25Okapi, BM25L, BM25Plus # 일단 Plus 만 사용 자세한 내용은 

import numpy as np

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)

import argparse
from train_dpr import Dense, BertEncoder, get_dense_args, preprocess
from transformers import AutoTokenizer


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

# Sparse Retrieval based on BM25
class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "/opt/ml/data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 BM25를 선언합니다.
        """
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # set 은 매번 순서가 바뀌므로
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        self.contexts =  list(map(preprocess,self.contexts))
        
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        self.BM25 = None
        self.tokenizer = tokenize_fn

    def get_sparse_embedding_bm25(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """
        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding_bm25.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.BM25 = pickle.load(file)
            print("BM25 Embedding pickle load.")

        else:
            print("Build passage BM25 embedding")
            tokenized = [self.tokenizer(i) for i in self.contexts]
            self.BM25 = BM25Plus(tokenized)

            with open(emd_path, "wb") as file:
                pickle.dump(self.BM25, file)
            print("BM25 Embedding pickle saved.")


    # 들어온 질문에 대해서 유사한 Text 찾음
    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc_BM25`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk_BM25`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.BM25 is not None, "get_sparse_embedding_BM25() 메소드를 먼저 수행"

        # 단일 query 에 대한 처리
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_BM25(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        # 여러 query 들에 대한 처리
        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_BM25(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval(BM25): ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas


    def get_relevant_doc_BM25(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        """
        tokenized_q = self.tokenizer(query)

        with timer("transform"):
            doc_scores = self.BM25.get_scores(tokenized_q)

        doc_indices = doc_scores.argmax()
        print(f'Document BM25 Scores: {doc_scores} \t Document Indices: {doc_indices}')
        return doc_scores, doc_indices



    def get_relevant_doc_bulk_BM25(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                Query들을 input으로 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            score과 indice들에 대한 pickle 파일 저장 혹은 불러오기 작업 수행
        """

        # 저장할 pickle 파일 및 경로 지정
        score_path = os.path.join(self.data_path, "BM25_score.bin")      
        indice_path = os.path.join(self.data_path, "BM25_indice.bin")

        # Pickle 파일 존재 시에 불러오기
        if os.path.isfile(score_path) and os.path.isfile(indice_path):
            with open(score_path, "rb") as file:
                doc_scores = pickle.load(file)  
            with open(indice_path, "rb") as file:
                doc_indices= pickle.load(file)            
            print("BM25 pickle load.")

        # Pickle 파일 생성 전일 시에 생성
        else:
            print("Build BM25 pickle")
            tokenized_queries= [self.tokenizer(i) for i in queries]        
            doc_scores = []
            doc_indices = []

            # Top-k 개에 대한 score 및 indices append
            for i in tqdm(tokenized_queries):
                scores = self.BM25.get_scores(i)

                sorted_score = np.sort(scores)[::-1]
                sorted_id = np.argsort(scores)[::-1]

                doc_scores.append(sorted_score[:k])
                doc_indices.append(sorted_id[:k])

            # Pickle 파일 dump 
            with open(score_path, "wb") as file:
                pickle.dump(doc_scores, file)
            with open(indice_path, "wb") as file:
                pickle.dump(doc_indices, file)
            print("BM25 pickle saved.")        

        return doc_scores, doc_indices
    

class DenseRetrieval(Dense):
    def __init__(self, **kwargs):
        super(DenseRetrieval, self).__init__(**kwargs)

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert self.p_encoder and self.q_encoder is not None, "get_dense_encoders() 먼저 수행"

        if isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_dpr(
                    query_or_dataset["question"], k=topk)
                # )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense Retriever: ")
            ):
                
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                # print(tmp['id'], tmp['question'])#, tmp['context'])
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                
                total.append(tmp)

            cqas = pd.DataFrame(total)
            # cqas.to_csv('retrieved_contexts.csv') # if neccessary
            return cqas 

    def get_relevant_doc_bulk_dpr(
        self, queries, k= 1, args=None, p_encoder=None, q_encoder=None
    ):
        if args is None:
            args = self.args
        if p_encoder is None:
            p_encoder = self.p_encoder
        if q_encoder is None:
            q_encoder = self.q_encoder
        
        p_encoder.to('cuda')
        q_encoder.to('cuda')

        doc_scores = []
        doc_indices = []

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            p_embs = []
            for p in self.contexts:
                p_inputs = self.tokenizer(
                    p,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).to("cuda")
            
                p_emb = p_encoder(**p_inputs).to("cpu").numpy()
                p_embs.append(p_emb)
            p_embs = torch.Tensor(p_embs).squeeze()

            q_embs = []
            for q in queries:
                q_inputs = self.tokenizer(
                    q,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).to("cuda")

                q_emb = q_encoder(**q_inputs).to("cpu").numpy()
                q_embs.append(q_emb)
            q_embs = torch.Tensor(q_embs).squeeze()
            
        dot_prod = torch.matmul(q_embs,torch.transpose(p_embs,0,1))
        doc_scores, doc_indices = torch.sort(dot_prod, dim = 1, descending = True)


        return doc_scores[:,:k], doc_indices[:,:k]


if __name__ == "__main__":
    from transformers import HfArgumentParser
    from arguments import RetrieverArguments
    parser = HfArgumentParser(( RetrieverArguments))
    retriever_args= parser.parse_args_into_dataclasses()

    # Test
    org_dataset = load_from_disk("../data/train_dataset")
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)


    args, tokenizer, p_enc, q_enc = get_dense_args(retriever_args)
    datasets = full_ds
    retriever = DenseRetrieval(args=args,dataset=datasets, 
                        tokenizer=tokenizer,p_encoder=p_enc,q_encoder=q_enc)

    def retriever_prec_k(topk_list):
        result_dict = {}
        with timer("bulk query by exhaustive search"):
            for k in tqdm(topk_list):
                result_retriever = retriever.retrieve(full_ds,topk = k)
                correct = 0
                for index in range(len(result_retriever)):
                    if  result_retriever['original_context'][index] in result_retriever['context'][index]:
                        correct += 1
                result_dict[k] = correct/len(result_retriever)
                print(result_dict)
        return result_dict

    result = retriever_prec_k([1,3,5,10])

    print(result)

