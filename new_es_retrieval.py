import os
from subprocess import Popen, PIPE, STDOUT
import time

from elasticsearch import Elasticsearch

import json
import pandas as pd

from tqdm import tqdm

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever import ElasticsearchRetriever
from haystack.pipeline import DocumentSearchPipeline

from datasets import load_from_disk


def set_index_and_server(args) :
    es_server = Popen([args.path_to_elastic],
                    stdout=PIPE, stderr=STDOUT,
                    preexec_fn=lambda: os.setuid(1)  # as daemon
                    )
    time.sleep(30)

    config = {'host':'localhost', 'port':9200}
    es = Elasticsearch([config])

    index_config = {
        "settings": {
            "analysis": {
                "filter":{
                    "my_stop_filter": {
                        "type" : "stop",
                        "stopwords_path" : "user_dic/my_stop_dic.txt"
                    }
                },
                "analyzer": {
                    "nori_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "decompound_mode": "mixed",
                        "filter" : ["my_stop_filter"]
                    }
                }
            }
        },
        "mappings": {
            "dynamic": "strict", 
            "properties": {
                "document_text": {"type": "text", "analyzer": "nori_analyzer"}
                }
            }
        }

    print('elastic serach ping :', es.ping())
    print(es.indices.create(index=args.index_name, body=index_config, ignore=400))
    return es

def set_datas() :
    # if not os.path.isfile("../data/preprocess_train.pkl") :
    #     make_custom_dataset("../data/preprocess_train.pkl")
#     train_file = load_from_disk("../data/train_dataset")["train"]
#     validation_file = load_from_disk("../data/train_dataset")["validation"]
    train_file = load_from_disk("../data/train_dataset")["train"]
    validation_file = load_from_disk("../data/train_dataset")["validation"]

    #[wiki-index, wiki-index-split-400, wiki-index-split-800, wiki-index-split-1000]
    
    dataset_path = "../data/wikipedia_documents.json"
        
    # if not os.path.isfile(dataset_path) :
    #     print(dataset_path)
    #     make_custom_dataset(dataset_path)

    with open(dataset_path, "r") as f:
        wiki = json.load(f)
    wiki_contexts = list(dict.fromkeys([v['text'] for v in wiki.values()]))

    qa_records = [{"example_id" : train_file[i]["id"], "document_title" : train_file[i]["title"], "question_text" : train_file[i]["question"], "answer" : train_file[i]["answers"]} for i in range(len(train_file))]
    wiki_articles = [{"document_text" : wiki_contexts[i]} for i in range(len(wiki_contexts))]
    return qa_records, wiki_articles


def populate_index(es_obj, index_name, evidence_corpus):

    for i, rec in enumerate(tqdm(evidence_corpus)):
        try:
            index_status = es_obj.index(index=index_name, id=i, body=rec)
        except:
            print(f'Unable to load document {i}.')
            
    n_records = es_obj.count(index=index_name)['count']
    print(f'Succesfully loaded {n_records} into {index_name}')



def es_retrieval():
    es_server = Popen(['elasticsearch-7.9.2/bin/elasticsearch'],
                    stdout=PIPE, stderr=STDOUT,
                    preexec_fn=lambda: os.setuid(1)  # as daemon
                    )
    print("wait until ES has started")
    time.sleep(30)
    print("end until ES has started")   

    es = Elasticsearch('localhost:9200')
    index_name  = "wiki-index"

    # with open('/opt/ml/data/wikipedia_documents.json', 'r') as f:
    #     wiki_data = pd.DataFrame(json.load(f)).transpose()

    # es.indices.delete(index="document", ignore=[400, 404]) # 하는 이유가?

    # mapping = {
    #     "settings": {
    #         "analysis": {
    #             "filter":{
    #                 "my_stop_filter": {
    #                     "type" : "stop",
    #                     "stopwords_path" : "user_dic/my_stop_dic.txt"
    #                 }
    #             },
    #             "analyzer": {
    #                 "nori_analyzer": {
    #                     "type": "custom",
    #                     "tokenizer": "nori_tokenizer",
    #                     "decompound_mode": "mixed",
    #                     "filter" : ["my_stop_filter"]
    #                 }
    #             }
    #         },
    #         'similarity':{
    #             'my_similarity':{
    #                 'type':'BM25',
    #             }
    #         }
    #     },
    #     "mappings": {
    #         "dynamic": "strict", 
    #         "properties": {
    #             "document_text": {"type": "text", "analyzer": "nori_analyzer"}
    #             }
    #         }
    #     }

    
    mapping = {
                      'settings':{
                          'analysis':{
                              'analyzer':{
                                  'my_analyzer':{
                                      "type": "custom",
                                      'tokenizer':'nori_tokenizer',
                                      'decompound_mode':'mixed',
                                      'stopwords':'_korean_',
                                      "filter": ["lowercase",
                                                 "my_shingle_f",
                                                 "nori_readingform",
                                                 "nori_number"]
                                  }
                              },
                              'filter':{
                                  'my_shingle_f':{
                                      "type": "shingle"
                                  }
                              }
                          },
                          'similarity':{
                              'my_similarity':{
                                  'type':'BM25',
                              }
                          }
                      },
                      'mappings':{
                          'properties':{
                              'title':{
                                  'type':'text',
                                  'analyzer':'my_analyzer',
                                  'similarity':'my_similarity'
                              },
                              'text':{
                                  'type':'text',
                                  'analyzer':'my_analyzer',
                                  'similarity':'my_similarity'
                              }
                          }
                      }
                  }

    print(es.indices.create(index=index_name, body=mapping, ignore=400))

    print("------------document_store------------")  
    # Connect to Elasticsearch
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document", custom_mapping=mapping)

    with open('/opt/ml/data/wikipedia_documents.json', "r") as f:
        wiki = json.load(f)
        
    # contexts = list(dict.fromkeys([v['text'] for v in wiki.values()]))
    contexts = list(dict.fromkeys([v["title"] + ": " + v["text"] for v in wiki.values()]))

    dicts = [
        {
            'text': context,
            #'meta': {}
        } for context in tqdm(contexts)
    ]

    # Now, let's write the docs to our DB.
    document_store.write_documents(dicts)


    retriever = ElasticsearchRetriever(document_store)
    pipe = DocumentSearchPipeline(retriever)

    testset=load_from_disk('/opt/ml/data/test_dataset')
    testset=testset['validation']

    # save_score={}

    # print("------------save_scores------------")  
    # for idx, example in enumerate(tqdm(testset, desc="elasticsearch: ")):
    #     # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
    #     question=example["question"]
    #     top_k_docs = pipe.run(question, params={"retriever": {"top_k": 10}})

    #     query = {
    #         'query':{
    #             'bool':{
    #                 'must':[
    #                         {'match':{'text':question}}
    #                 ],
    #                 'should':[
    #                         {'match':{'text':question}}
    #                 ]
    #             }
    #         }
    #     }
    #     doc = es.search(index='document',body=query,size=30)['hits']['hits']

        
    #     save_score[example['id']]=doc[0]['_score']
        
    # with open('top1_score.json', "w") as writer:
    #     writer.write(json.dumps(save_score, indent=4, ensure_ascii=False) + "\n")

    total = []
    print("------------save_doc------------")  
    for idx, example in enumerate(tqdm(testset, desc="elasticsearch: ")):
        # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
        question=example["question"]
        top_k_docs = pipe.run(question, params={"retriever": {"top_k": 10}})

        query = {
            'query':{
                'bool':{
                    'must':[
                            {'match':{'text':question}}
                    ],
                    'should':[
                            {'match':{'text':question}}
                    ]
                }
            }
        }

        # query = {
        #     'query': {
        #         'match': {
        #             'document_text': question
        #             }
        #         }
        #     }

        doc = es.search(index='document',body=query,size=30)['hits']['hits']
        # print(doc)
        # doc = es.search(index=index_name, body=query, size=n_results)
        cc = ''

        for i in range(5):
            cc += doc[i]['_source']['text']

        tmp = {
            "question": example["question"],
            "id": example['id'],
            "context_id": doc[0]['_id'],  # retrieved id
            "context": cc  # retrieved doument
        }
    
        if 'context' in example.keys() and 'answers' in example.keys():
            tmp["original_context"] = example['context']  # original document
            tmp["answers"] = example['answers']           # original answer
        total.append(tmp)

    print("finish_retrieve")
    cqas = pd.DataFrame(total)
    cqas.to_csv('question1_contest.csv')

    return cqas


# def main(args) :
#     print('Start to Set Elastic Search')
#     _, wiki_articles = set_datas(args)
#     es = set_index_and_server(args)
#     populate_index(es_obj=es, index_name="wiki-index", evidence_corpus=wiki_articles)
#     print('Finish')
