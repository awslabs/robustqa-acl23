'''This script converts standard data: passages.jsonl + qrels.jsonl into BEIR format'''

from argparse import ArgumentParser
import json
import os
import pandas as pd


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--split', default='test', type=str)
    args = parser.parse_args()

    if args.data in ['lifestyle', 'recreation', 'technology', 'science', 'writing']:
        root = f'{args.data_dir}/{args.data}/{args.split}'
    else:
        root = f'{args.data_dir}/{args.data}'


    passages = []
    with open(f'{root}/passages.jsonl') as infile: 
        for line in infile:
            sample = json.loads(line)
            sample["_id"] =  f"doc{sample['pid']}"
            del sample['pid']
            passages.append(sample)

    with open(f'{root}/corpus.jsonl', 'w') as outfile:
        for sample in passages:
            json.dump(sample, outfile)
            outfile.write('\n')


    annotations, all_qids = [], []
    with open(f'{root}/qrels.jsonl') as infile: 
        for r, line in enumerate(infile):
            sample = json.loads(line)
            annotations.append({"_id": f'{args.split}{r}',
                                "text": sample['question'],
                                "metadata": {}})
            all_qids.append(f'test{r}')

    with open(f'{root}/queries.jsonl', 'w') as outfile:
        for sample in annotations:
            json.dump(sample, outfile)
            outfile.write('\n')


    # placeholder file to run bm25
    qrel_path = f"{root}/qrels/"
    if not os.path.exists(qrel_path):
        os.mkdir(qrel_path)

    N = len(all_qids)
    qrel = pd.DataFrame(data={"query-id": all_qids, "corpus-id": ["doc0" for i in range(N)], "score": [0]*N})
    qrel.to_csv(qrel_path + "%s.tsv" % args.split, sep='\t', index=False)