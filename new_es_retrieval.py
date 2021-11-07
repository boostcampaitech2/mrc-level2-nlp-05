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


def es_retrieval():
    es_server = Popen(['elasticsearch-7.9.2/bin/elasticsearch'],
                    stdout=PIPE, stderr=STDOUT,
                    preexec_fn=lambda: os.setuid(1)  # as daemon
                    )
    print("wait until ES has started")
    time.sleep(30)
    print("end until ES has started")   

    es = Elasticsearch('localhost:9200')

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
                                  'type':'IB',
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

        doc = es.search(index='document',body=query,size=30)['hits']['hits']

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
    cqas.to_csv('context_IB.csv')

    return cqas
