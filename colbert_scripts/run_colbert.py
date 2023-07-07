import os
import sys
from pathlib import Path
sys.path.insert(0, '../')

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

import json
import pandas as pd
import ast
import argparse
from tqdm import tqdm
from collections import OrderedDict

from qa_validation import calculate_matches

def get_all_answers(datadir):
    answers, queries = [], []
    with open(f'{datadir}qrels.jsonl') as infile:
        for line in infile:
            ex = json.loads(line)
            queries.append(ex['question'])
            answers.append(ex['answers'])
    assert len(answers) == len(queries)
    return answers

def get_top_passages(pr_results, answers, queries, k=100, tk=100):
    all_returns = []
    qid = None
    pids, scores = [], []
    for r, row in pr_results.iterrows():
        qi = str(int(row['qid']))
        pi = str(int(row['pid']))
        if qi != qid:
            assert (len(pids) == 0 or len(pids) == k)
            if len(pids) == k:
                all_returns.append((pids[:tk], scores[:tk]))
            qid = qi
            pids = [pi]
            scores = [row['score']]
        else:
            pids.append(pi)
            scores.append(row['score'])

    assert len(pids) == k
    all_returns.append((pids[:tk], scores[:tk]))

    print(len(all_returns), len(answers), len(queries))

    return all_returns

def save_pr_results(match_stats, top_passages, all_answers, all_passages, all_queries, out_file):
    assert len(match_stats) == len(top_passages) == len(all_answers) == len(all_queries)
    all_samples = []
    with open(out_file, 'w') as outfile:
        for q, (hits, tp, ans, query) in enumerate(zip(match_stats, top_passages, all_answers, all_queries)):
            assert len(hits) == len(tp[0])
            ctxs = []
            for i, hit in enumerate(hits):
                pid = tp[0][i]
                passage = {"id": pid,
                           "title": all_passages[pid][0],
                           "text": all_passages[pid][1],
                           "score": tp[1][i],
                           "has_answer": hit}
                ctxs.append(passage)
            json.dump({"query": query, "answers": ans, "passages": ctxs}, outfile)
            outfile.write('\n')

            all_samples.append({"question": query,
                                "answers": ans,
                                "ctxs": ctxs})

    print(len(all_samples))

    with open(out_file.replace('jsonl', 'json'), "w") as writer:
        writer.write(json.dumps(all_samples, indent=4) + "\n")
    print("Saved results * scores  to %s", out_file)
    return

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--dataset",
                        default='nq',
                        type=str,
                        help="data")
    parser.add_argument("--dataroot",
                        default='',
                        type=str,
                        help="data dir")
    parser.add_argument("--split",
                        default='test',
                        type=str,
                        help="data split")
    parser.add_argument("--model",
                        default='',
                        type=str,
                        help="model checkpoint")
    parser.add_argument("--reindex",
                        default=False,
                        action='store_true',
                        help="Re-index passages")
    parser.add_argument("--index",
                        default=False,
                        action='store_true',
                        help="Index passages")
    parser.add_argument("--search",
                        default=False,
                        action='store_true',
                        help="Retrieve passages")
    parser.add_argument("--eval",
                        default=False,
                        action='store_true',
                        help="Evaluate")
    parser.add_argument("--ranking_file",
                        default="",
                        help="ranked top passage file for evaluation")

    args = parser.parse_args()

    dataroot = args.dataroot
    dataset = args.dataset
    datasplit = args.split

    nbits = 2  # encode each dimension with 2 bits

    checkpoint = args.model
    index_name = f'{dataset}.{nbits}bits'

    collection = os.path.join(dataroot, dataset, 'collection.tsv')
    if datasplit == 'test':
        query_dir = os.path.join(dataroot, dataset, 'questions.tsv')
    else:
        query_dir = os.path.join(dataroot, dataset, 'questions.%s.tsv' % datasplit)

    if args.index:
        with Run().context(RunConfig(nranks=4, experiment=dataset)):
            config = ColBERTConfig(
                nbits=2,
            )
            indexer = Indexer(checkpoint=checkpoint, config=config)
            indexer.index(name=index_name, collection=collection, overwrite=args.reindex)

    if args.search:
        with Run().context(RunConfig(experiment=dataset)):
            queries = Queries(query_dir)
            searcher = Searcher(index=index_name)
            results = searcher.search_all(queries, k=100)
            save_file = f"{args.dataset}-{datasplit}-ranking.tsv"
            results.save(save_file)

    if args.eval:
        datadir = f'{dataroot}{dataset}/'

        results = pd.read_csv(args.ranking_file, sep='\t', header=None)
        results.columns = ['qid', 'pid', 'rank', 'score']

        queries = pd.read_csv(query_dir, sep='\t', header=None)
        queries.columns = ['qid', 'text']

        all_queries = OrderedDict([(row['qid'], row['text'].lower()) for _, row in queries.iterrows()])

        all_answers = get_all_answers(datadir)

        top_passages = get_top_passages(results, all_answers, all_queries)

        assert len(all_answers) == len(top_passages)

        all_passages = {}

        with open(f'{datadir}passages.jsonl') as infile:
            for i, line in enumerate(tqdm(infile)):
                sample = json.loads(line)
                all_passages[str(i)] = (sample['title'], sample['text'])

        print("\nTotal %s passages and  %s queries" % (len(all_passages), len(top_passages)))

        all_answers, top_passages = all_answers, top_passages
        print(len(all_answers), len(top_passages), len(all_queries), len(all_passages))

        match_stats = calculate_matches(all_passages, all_answers, top_passages, workers_num=16, match_type='string')

        top_k_hits = match_stats.top_k_hits

        print("Validation results: top k documents hits %s" % top_k_hits)
        top_k_hits = [v / len(top_passages) for v in top_k_hits]
        print("Validation results: top k documents hits accuracy %s" % top_k_hits)
        print(top_k_hits[4], top_k_hits[9], top_k_hits[19], top_k_hits[49], top_k_hits[99])

        outdir = f'output/'
        Path(outdir).mkdir(parents=True, exist_ok=True)
        outfile = f"{outdir}{dataset}_from_colbert_{args.split}.jsonl"
        save_pr_results(match_stats.questions_doc_hits, top_passages, all_answers, all_passages,
                        list(all_queries.values()), outfile)


if __name__=='__main__':
    main()