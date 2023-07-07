'''This script converts standard data: passages.jsonl + qrels.jsonl into DPR format'''

from argparse import ArgumentParser
import json
from tqdm import tqdm
import pandas as pd

def create_passages(root, dataset, output_dir):
    doc_ids, texts, titles = [], [], []
    with open(f'{root}/passages.jsonl') as infile:
        for i, line in enumerate(tqdm(infile)):
            sample = json.loads(line)
            doc_ids.append(sample['pid'])
            texts.append(sample['text'].replace('\n', ' '))
            titles.append(sample['title'])

    df = pd.DataFrame(list(zip(doc_ids, texts, titles)))
    df.columns = ['id', 'text', 'title']
    df.to_csv(f'{output_dir}/{dataset}_split/psgs_w100.tsv', sep='\t', index=False)

def create_qas(root, dataset, output_dir, split='test'):
    questions, answers = [], []
    with open(f'{root}/qrels.jsonl') as infile:
        for j, line in enumerate(tqdm(infile)):
            sample = json.loads(line)
            questions.append(sample['question'])
            answers.append(sample['answers'])
    df = pd.DataFrame(list(zip(questions, answers)))
    df.to_csv(f'{output_dir}/retriever/qas/{dataset}-{split}.csv', sep='\t', header=False, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--output_dir', default='dpr', type=str)
    args = parser.parse_args()

    # only LoTTE data has dev/test splits
    if args.data in ['lifestyle', 'recreation', 'technology', 'science', 'writing']:
        root = f'{args.data_dir}/{args.data}/{args.split}'
    else:
        root = f'{args.data_dir}/{args.data}'

    create_passages(root, args.data, args.output_dir)
    create_qas(root, args.data, args.output_dir, args.split)