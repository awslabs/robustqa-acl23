'''This script converts standard data: passages.jsonl + qrels.jsonl into ColBERTv2 format'''

from argparse import ArgumentParser
import json
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import os

def create_passages(root, dataset, output_dir):
    doc_ids, texts = [], []
    with open(f'{root}/passages.jsonl') as infile:
        for i, line in enumerate(tqdm(infile)):
            sample = json.loads(line)
            doc_ids.append(str(i))
            texts.append(sample['text'].replace('\n', ' '))

    df = pd.DataFrame(list(zip(doc_ids, texts)))

    outdir = f'{output_dir}/{dataset}/'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{outdir}collection.tsv', sep='\t', header=False, index=False)
    # additionally make a copy of passage file to the output dir for performance calculation
    os.system(f'cp {root}/passages.jsonl {outdir}')

def create_queries(root, dataset, output_dir):
    ids, texts = [], []
    qrels = []
    with open(f'{root}/qrels.jsonl') as infile:
        for j, line in enumerate(tqdm(infile)):
            sample = json.loads(line)
            qrels.append(sample)
            ids.append(str(j))
            texts.append(sample['question'].replace('\n', ' '))
    df = pd.DataFrame(list(zip(ids, texts)))

    outdir = f'{output_dir}/{dataset}/'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{outdir}questions.tsv', sep='\t', header=False, index=False)

    # additionally make a copy of qrel file to the output dir for performance calculation
    os.system(f'cp {root}/qrels.jsonl {outdir}')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--output_dir', default='colbert', type=str)
    args = parser.parse_args()

    # only LoTTE data has dev/test splits
    if args.data in ['lifestyle', 'recreation', 'technology', 'science', 'writing']:
        root = f'{args.data_dir}/{args.data}/{args.split}'
    else:
        root = f'{args.data_dir}/{args.data}'

    create_passages(root, args.data, args.output_dir)
    create_queries(root, args.data, args.output_dir)
