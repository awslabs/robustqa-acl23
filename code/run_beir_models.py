from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from beir import LoggingHandler
import logging


from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval

from beir.reranking.models import CrossEncoder

from reranker import Rerank
from qa_validation import calculate_matches
import pandas as pd
import ast
import argparse
import json

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def eval_perf_bm25_ce(dataset, datadir, split, reindex=False, model_dir="cross-encoder/ms-marco-electra-base"):
    corpus, queries, qrels = GenericDataLoader(datadir).load(split=split)
    all_passages = get_all_passages(corpus)

    all_answers = get_all_answers(datadir)

    #### Provide parameters for elastic-search
    hostname = "localhost"
    index_name = dataset

    model = BM25(index_name=index_name, hostname=hostname, initialize=reindex)
    retriever = EvaluateRetrieval(model)

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    cross_encoder_model = CrossEncoder(model_dir)
    reranker = Rerank(cross_encoder_model, batch_size=128)

    # Rerank top-100 results using the reranker provided
    results = reranker.rerank(corpus, queries, results, top_k=100)
    top_passages, all_answers, all_queries = get_top_passages(results, all_answers, queries)

    assert len(all_answers) == len(top_passages)
    print("\nTotal %s passages and  %s queries" % (len(all_passages), len(top_passages)))
    match_stats = calculate_matches(all_passages, all_answers, top_passages, workers_num=8, match_type='string')

    top_k_hits = match_stats.top_k_hits

    print("Validation results: top k documents hits %s" % top_k_hits)
    top_k_hits = [v / len(top_passages) for v in top_k_hits]
    print("Validation results: top k documents hits accuracy %s" % top_k_hits)
    print(top_k_hits[4], top_k_hits[9], top_k_hits[19], top_k_hits[49], top_k_hits[-1])

    output_dir = f"output/{dataset}/{split}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    outfile = f"{output_dir}/{dataset}_from_bm25_ce_{split}.json"

    save_pr_results(match_stats.questions_doc_hits, top_passages, all_answers, all_passages, all_queries, outfile)


def eval_perf_bm25(dataset, datadir, split, reindex):
    corpus, queries, qrels = GenericDataLoader(datadir).load(split=split)

    all_passages = get_all_passages(corpus)

    all_answers = get_all_answers(datadir)

    #### Provide parameters for elastic-search
    hostname = "localhost"
    index_name = dataset

    model = BM25(index_name=index_name, hostname=hostname, initialize=reindex)
    retriever = EvaluateRetrieval(model)

    #### Retrieve passages
    results = retriever.retrieve(corpus, queries)
    top_passages, all_answers, all_queries = get_top_passages(results, all_answers, queries)

    assert len(all_answers) == len(top_passages)
    print("\nTotal %s passages and  %s queries" % (len(all_passages), len(top_passages)))
    match_stats = calculate_matches(all_passages, all_answers, top_passages, workers_num=8, match_type='string')

    top_k_hits = match_stats.top_k_hits

    print("Validation results: top k documents hits %s" % top_k_hits)
    top_k_hits = [v / len(top_passages) for v in top_k_hits]
    print("Validation results: top k documents hits accuracy %s" % top_k_hits)
    print(top_k_hits[4], top_k_hits[9], top_k_hits[19], top_k_hits[49], top_k_hits[-1])

    output_dir = f"output/{dataset}/{split}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    outfile = f"{output_dir}/{dataset}_from_bm25_ce_{split}.json"

    save_pr_results(match_stats.questions_doc_hits, top_passages, all_answers, all_passages, all_queries, outfile)

def get_top_passages(all_results, all_answers, all_queries, topk=100):

    # for some reasons, some queries are not returned
    all_returns, filtered_answers, filtered_queries = [], [], []
    for k, v in all_results.items():
        idx = int(k[4:])
        filtered_answers.append(all_answers[idx])
        filtered_queries.append(all_queries[k])
        results = sorted([(s, i) for i, s in v.items()], key=lambda x: -x[0])[:topk]
        pids = [x[1] for x in results]
        scores = [x[0] for x in results]
        all_returns.append((pids, scores))

    print("\nFilter out %s queries" % (len(all_answers) - len(filtered_answers)))
    assert len(all_returns) == len(filtered_queries) == len(filtered_answers)
    return all_returns, filtered_answers, filtered_queries

def get_all_passages(corpus):
    all_passages = {}
    for k, v in corpus.items():
        all_passages[k] = (v['text'], v['title'])
    return all_passages

def get_all_answers(datadir):

    all_answers = []
    with open(f'{datadir}/qrels.jsonl') as infile:
        for line in infile:
            ex = json.loads(line)
            all_answers.append(ex['answers'])

    return all_answers


def save_pr_results(match_stats, top_passages, all_answers, all_passages, all_queries, out_file, doc_id="wiki:"):
    assert len(match_stats) == len(top_passages) == len(all_answers) == len(all_queries)
    all_samples = []
    for q, (hits, tp, ans, query) in enumerate(zip(match_stats, top_passages, all_answers, all_queries)):
        assert len(hits) == len(tp[0])
        ctxs = []
        for i, hit in enumerate(hits):
            pid = tp[0][i]
            passage = {"id": pid.replace('doc', doc_id),
                       "title": all_passages[pid][1],
                       "text": all_passages[pid][0],
                       "score": tp[1][i],
                       "has_answer": hit}
            ctxs.append(passage)
        all_samples.append({"question": query,
                            "answers": ans,
                            "ctxs": ctxs})

    with open(out_file, "w") as writer:
        writer.write(json.dumps(all_samples, indent=4) + "\n")
    print("Saved results * scores  to %s", out_file)
    return all_samples


def main(args):

    dataset = args.data

    if args.data in ['lifestyle', 'recreation', 'technology', 'science', 'writing']:
        root = f'{args.data_dir}/{args.data}/{args.split}'
    else:
        root = f'{args.data_dir}/{args.data}'

    if args.model == 'bm25':
        eval_perf_bm25(dataset, root, args.split, args.reindex)
    elif args.model == 'bm25+ce':
        eval_perf_bm25_ce(dataset, root, args.split, args.reindex)
    else:
        print("Model not supported!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default=None,
                        type=str,
                        required=True,
                        help="Target data")
    parser.add_argument("--data_dir",
                        default='data',
                        type=str,
                        help="input data directory")
    parser.add_argument("--split",
                        default="test",
                        type=str,
                        help="Data split")
    parser.add_argument("--reindex",
                        default=False,
                        action='store_true',
                        help="Re-index passages")
    parser.add_argument("--model",
                        default="bm25",
                        required=True,
                        help="bm25 or bm25+ce")
    
    args = parser.parse_args()
    main(args)

