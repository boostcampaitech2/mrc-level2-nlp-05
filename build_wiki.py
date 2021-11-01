import os
from collections import defaultdict
from typing import Dict, Optional
from datasets.load import load_from_disk

import pandas as pd

from transformers import AutoTokenizer
from datasets import Dataset


def load_wiki_from_json(
    wiki_path: str = "/opt/ml/data/wikipedia_documents.json"
) -> Dataset:

    # load json as a pandas DataFrame first for transposing
    print("Loading dataset as pandas")
    wiki_df = pd.read_json(wiki_path).T
    
    wiki_dataset = Dataset.from_pandas(wiki_df)

    return wiki_dataset

def _split_into_chunks(
    examples, 
    max_chunk_len: int = 500, 
    doc_stride: int = 100
) -> Dict:
    
    outputs = defaultdict(list)

    column_names = list(examples.keys())
    column_names.remove("text")

    texts = examples['text']

    for example_idx, text in enumerate(texts):
        split_ids = list(range(0, len(text), max_chunk_len-doc_stride)) + [len(text)]

        num_chunks = 0
        for i in range(len(split_ids)-1):
            start_idx, end_idx = split_ids[i], min(split_ids[i+1]+doc_stride, len(text))
            if end_idx - start_idx < doc_stride:
                # already fully included in the previous chunk!
                # therefore, safe to break out the loop!
                break

            split_text = text[start_idx:end_idx]
            split_text = split_text.strip()
            outputs['text'].append(split_text)
            num_chunks += 1
        
        for col in column_names:
            outputs[col].extend([examples[col][example_idx]] * num_chunks)

    return outputs

def split_into_chunks(dataset: Dataset) -> Dataset:
    return dataset.map(_split_into_chunks, batched=True, num_proc=4)

def _add_title_to_text(
    example, 
    sep: str=":"
) -> Dict:

    if sep[-1] != " ":
        # insert a space
        sep = sep + " "

    prefix = example['title'] + sep

    return {"text": prefix + example["text"]}

def add_title_to_texts(dataset: Dataset) -> Dataset:
    return dataset.map(_add_title_to_text, batched=False, num_proc=4)

def _split_into_sentences(
    examples, 
    max_chunk_len: int = 500
) -> Dict:
    outputs = defaultdict(list)

    column_names = list(examples.keys())
    column_names.remove("text")

    texts = examples['text']

    for example_idx, text in enumerate(texts):
        subtexts = [subtext for subtext in text.split("\n") if len(subtext) > 0]

        current_context = ""
        num_chunks = 0

        for subtext in subtexts:
            if len(current_context) + len(subtext) < max_chunk_len:
                current_context += subtext
            else:
                outputs['text'].append(current_context)
                num_chunks += 1
                current_context = subtext

        # extreme case
        if ("text" not in outputs) or (current_context != outputs['text'][-1]):
            outputs['text'].append(current_context)
            num_chunks += 1

        for col in column_names:
            outputs[col].extend([examples[col][example_idx]] * num_chunks)
    
    return outputs

def split_into_sentences(dataset: Dataset) -> Dataset:
    return dataset.map(_split_into_sentences, batched=True, num_proc=4)

def build_wiki(wiki_path: str, save_path: Optional[str] = None):
    """Builds wiki dataset from json file. 
    Also, saves as a HuggingFace's arrow dataset if `save_path` is provided."""

    wiki_dataset = load_wiki_from_json(wiki_path)
    print("wiki_dataset loaded")

    # wiki_dataset = split_into_sentences(wiki_dataset)
    # print("wiki_dataset splited into sentences")
    # print(wiki_dataset)

    # wiki_dataset = add_title_to_texts(wiki_dataset)
    # print("wiki_dataset title added")
    # print(wiki_dataset)

    if save_path is not None:
        wiki_dataset.save_to_disk(save_path)

    return wiki_dataset

def get_wiki(wiki_path: str, save_path: Optional[str] = None) -> Dataset:
    """This is a helper function that can be imported and used in other scripts.
    This function returns the dataset built from `build_wiki()` function."""
    if os.path.isdir(save_path) and len(os.listdir(save_path)) > 0:
        print("Wiki dataset already built in save_path. End the script.")
        wiki_dataset = load_from_disk(save_path)
        return wiki_dataset
    else:
        wiki_dataset = build_wiki(wiki_path=wiki_path, save_path=save_path)
        return wiki_dataset

def main():

    wiki_path = "/opt/ml/data/wikipedia_documents.json"
    save_path = "/opt/ml/data/wiki_data"

    get_wiki(wiki_path, save_path)

if __name__ == "__main__":
    main()
