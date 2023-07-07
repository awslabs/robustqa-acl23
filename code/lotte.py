import json
import logging
from processor import DataProcessor
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class LoTTE(DataProcessor):

    def process_documents(self):

        collection = []
        with open(f'{self.root}/collection.tsv') as f:
            for line_idx, line in enumerate(f):

                pid, passage, *rest = line.strip('\n\r ').split('\t')
                assert pid == 'id' or int(pid) == line_idx
                
                title = rest[0] if len(rest) >= 1 else ''

                collection.append({'doc_id': pid, 'title': title, 'text': passage, 'metadata': {}})

        with open(f'{self.root}/documents.jsonl', 'w') as outfile:
            for doc in collection:
                json.dump(doc, outfile)
                outfile.write('\n')
        return
    
    def load_questions(self):
        qid_to_question = {}

        dataset = self.root.split('/')[1]
        split = self.root.split('/')[-1]
        questions = pd.read_csv(f'{self.root}/questions.forum.tsv', sep='\t', header=None)
        questions.columns = ['r', 'question']
        for r, question in questions.iterrows():
            qid = f'{dataset}-forum-{split}-{r}'
            qid_to_question[qid] = question['question']

        questions = pd.read_csv(f'{self.root}/questions.search.tsv', sep='\t', header=None)
        questions.columns = ['r', 'question']
        for r, question in questions.iterrows():
            qid = f'{dataset}-search-{split}-{r}'
            qid_to_question[qid] = question['question']

        return qid_to_question

    def load_documents(self):
        docid_to_document = {}

        with open(f'{self.root}/collection.tsv') as f:
            for line_idx, line in enumerate(f):
                pid, passage, *rest = line.strip('\n\r ').split('\t')
                assert pid == 'id' or int(pid) == line_idx
                docid_to_document[pid] = passage       

        return docid_to_document
