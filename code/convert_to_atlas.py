'''This script converts standard data: passages.jsonl + qrels.jsonl into Atlas format'''

from argparse import ArgumentParser
import json
from pathlib import Path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--output_dir', default='atlas', type=str)
    args = parser.parse_args()


    if args.data in ['lifestyle', 'recreation', 'technology', 'science', 'writing']:
        root = f'{args.data_dir}/{args.data}/{args.split}'
    else:
        root = f'{args.data_dir}/{args.data}'

    outdir = f'{args.output_dir}/'
    Path(outdir).mkdir(parents=True, exist_ok=True)

    with open(f'{outdir}/{args.data}-test.jsonl', 'w') as outfile:
        with open(f'{root}/qrels.jsonl') as infile:
            for line in infile:
                sample = json.loads(line)
                question = sample['question']
                answers = sample['answers']
                entry = {'question': question, 'answers': answers}
                json.dump(entry, outfile)
                outfile.write('\n')


    with open(f'{outdir}/{args.data}-passages.jsonl', 'w') as outfile:
        with open(f'{root}/qrels.jsonl') as infile:
            for line in infile:
                sample = json.loads(line)
                entry = {'id': str(sample['pid']), 'title': sample['title'], 'text': sample['text']}
                json.dump(entry, outfile)
                outfile.write('\n')