import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch

from rank_bm25 import BM25Okapi, BM25L, BM25Plus 

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)

from sklearn.feature_extraction.text import TfidfVectorizer

from train_dpr import Dense, BertEncoder, get_dense_args, preprocess
from transformers import AutoTokenizer, HfArgumentParser
from arguments import RetrieverArguments


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


# Sparse Retrieval based on BM25
class SparseRetrieval_BM25P:
    """Passage 파일을 불러오고 BM25를 선언"""
    def __init__(self, tokenize_fn,data_path: Optional[str] = "/opt/ml/data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # set 은 매번 순서가 바뀌므로
        self.contexts = list(dict.fromkeys([v["title"] + ": " + v["text"] for v in wiki.values()]))

        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        self.BM25 = None
        self.tokenizer = tokenize_fn

    def get_sparse_embedding_bm25(self) -> NoReturn:
        """Create or import embeddings"""

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


    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """Query들에 대해서 retrieved 된 Passage 반환"""

        assert self.BM25 is not None, "get_sparse_embedding_BM25() 메소드를 먼저 수행"

        if isinstance(query_or_dataset, Dataset):
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
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx],
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc_bulk_BM25(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """top k 개의 score&indice들에 대한 pickle 파일 저장 혹은 불러오기 작업 수행"""

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

class SparseRetrieval_TFIDF:
    def __init__(self, tokenize_fn,data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json") -> NoReturn:

        """Passage 파일을 불러오고 TfidfVectorizer를 선언"""

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # set 은 매번 순서가 바뀌므로
        self.contexts = list(dict.fromkeys([v["title"] + ": " + v["text"] for v in wiki.values()]))

        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.p_embedding = None 

    def get_sparse_embedding(self) -> NoReturn:
        """Create or import embeddings"""

        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")


    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """Query들에 대해서 retrieved 된 Passage 반환"""

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."


        if isinstance(query_or_dataset, Dataset):

            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # if validation set
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas


    def get_relevant_doc_bulk( self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """top k 개의 score&indice들에 대한 pickle 파일 저장 혹은 불러오기 작업 수행"""

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
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
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense Retriever: ")
            ):
                
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx],
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # if validation set
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                
                total.append(tmp)

            cqas = pd.DataFrame(total)
            # cqas.to_csv('retrieved_contexts.csv') # if neccessary
            return cqas 

    def get_relevant_doc_bulk_dpr(
        self, queries, k= 1, args=None, p_encoder=None, q_encoder=None
    ):
        """top k 개의 score & indice를 반환"""
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


# measuring topk retrieval performance
if __name__ == "__main__":

    parser = HfArgumentParser((RetrieverArguments))
    retriever_args= parser.parse_args_into_dataclasses()

    # Test
    org_dataset = load_from_disk("/opt/ml/data/train_dataset/")
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    args, tokenizer, p_enc, q_enc = get_dense_args(retriever_args)
    retriever = DenseRetrieval(args=args,dataset=full_ds, 
                        tokenizer=tokenizer,p_encoder=p_enc,q_encoder=q_enc)

    # tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    # retriever = SparseRetrieval(
    # tokenize_fn=tokenizer.tokenize, data_path="/opt/ml/data/", context_path="wikipedia_documents.json"
    # ) 
    # retriever.get_sparse_embedding_bm25()

    # df =  retriever.retrieve(full_ds,topk = 10)

    def retriever_prec_k(topk_list):
        result_dict = {}
        with timer("bulk query by exhaustive search"):
            for k in tqdm(topk_list):
                result_retriever = retriever.retrieve(full_ds,topk = k)
                correct = 0
                for index in range(len(result_retriever)):
                    if result_retriever['original_context'][index] in result_retriever['context'][index]:
                        correct += 1
                result_dict[k] = correct/len(result_retriever)
                print(result_dict)
        return result_dict

    result = retriever_prec_k([1,3,5,10])

    print(result)