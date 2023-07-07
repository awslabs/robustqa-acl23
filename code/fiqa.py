import json
import logging
from processor import DataProcessor
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



class FiQADataProcessor(DataProcessor):
    '''Only use train split for FiQA as test split doesn't have answer groundtruth'''
    def process_documents(self):

        documents = pd.read_csv(f'{self.root}/FiQA_train_doc_final.tsv', sep='\t', index_col=0)
        with open(f'{self.root}/documents.jsonl', 'w') as outfile:
            for r, row in documents.iterrows():
                json.dump({'doc_id': row['docid'], 'title': '', 'text': str(row['doc']), 'metadata': {'timestamp': row['timestamp']}}, outfile)
                outfile.write('\n')
        return
    
    def load_questions(self):
        qid_to_question = {}

        questions = pd.read_csv(f'{self.root}/FiQA_train_question_final.tsv', sep='\t')
        for _, question in questions.iterrows():
            qid = question['qid']
            qid_to_question[qid] = question['question']
        return qid_to_question

    def load_documents(self):
        docid_to_document = {}
        documents = pd.read_csv(f'{self.root}/FiQA_train_doc_final.tsv', sep='\t')
        for _, doc in documents.iterrows():
            docid = str(doc['docid'])
            docid_to_document[docid] = doc['doc']
        return docid_to_document
    
