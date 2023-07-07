from argparse import ArgumentParser
import pandas as pd
import json
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('--data', required=True, type=str)
parser.add_argument('--data_dir', default='final_data', type=str)
parser.add_argument('--split', default='test', type=str)
args = parser.parse_args()

def load_questions(data_dir):
    qid_to_question = {}

    if 'fiqa' in data_dir:
        questions = pd.read_csv(f'{data_dir}/FiQA_train_question_final.tsv', sep='\t')
        for _, question in questions.iterrows():
            qid = question['qid']
            qid_to_question[qid] = question['question']
    else:
        dataset = data_dir.split('/')[1]
        split = data_dir.split('/')[-1]
        questions = pd.read_csv(f'{data_dir}/questions.forum.tsv', sep='\t', header=None)
        questions.columns = ['r', 'question']
        for r, question in questions.iterrows():
            qid = f'{dataset}-forum-{split}-{r}'
            qid_to_question[qid] = question['question']

        questions = pd.read_csv(f'{data_dir}/questions.search.tsv', sep='\t', header=None)
        questions.columns = ['r', 'question']
        for r, question in questions.iterrows():
            qid = f'{dataset}-search-{split}-{r}'
            qid_to_question[qid] = question['question']

    return qid_to_question

def load_documents(data_dir):
    docid_to_document = {}
    if 'fiqa' in data_dir:
        documents = pd.read_csv(f'{data_dir}/FiQA_train_doc_final.tsv', sep='\t')
        for _, doc in documents.iterrows():
            docid = str(doc['docid'])
            docid_to_document[docid] = doc['doc']
    else:
        with open(f'{data_dir}/collection.tsv') as f:
            for line_idx, line in enumerate(f):
                pid, passage, *rest = line.strip('\n\r ').split('\t')
                assert pid == 'id' or int(pid) == line_idx
                docid_to_document[pid] = passage       

    return docid_to_document

# only LoTTE data has dev/test splits
if args.data in ['lifestyle', 'recreation', 'technology', 'science', 'writing']:
    root = f'{args.data_dir}/{args.data}/{args.split}'
else:
    root = f'{args.data_dir}/{args.data}'


data_file = f'{root}/samples.csv'
samples = pd.read_csv(data_file, keep_default_na=False)    

qid_to_question = load_questions(root)
doc_id_to_doc = load_documents(root)

new_samples = []
for r, row in samples.iterrows():
    qid = row['qid']
    docid = str(row['docid'])

    if row['query'] == 'What economic, political and other factors influence mortgage rates (and how)?':
        qid = 2361


    if row['query'] not in qid_to_question[qid]:
        print(qid)
        print(row['query'], len(row['query']))
        print(qid_to_question[qid], len(qid_to_question[qid]))
    if row['doc'] not in doc_id_to_doc[docid]:
        print(docid)
        print(row['doc'])
        print(doc_id_to_doc[docid])
        print("=" * 100)

    new_samples.append({'qid': qid, 'docid': docid, 'answer1': row['Answer span 1'], 'answer2': row['Answer span 2'], 'answer3': row['Answer span 3']})

save_dir = root.replace('final_', '')
Path(save_dir).mkdir(parents=True, exist_ok=True)
with open(f'{save_dir}/samples.jsonl', 'w') as outfile:
    for sample in new_samples:
        json.dump(sample, outfile)
        outfile.write('\n')