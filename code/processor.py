import json
import logging
import numpy as np
from collections import defaultdict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor():

    def __init__(self, root):
        self.root = root

    def load_questions(self):
        pass

    def load_documents(self):
        pass

    def process_documents(self):
        pass

    def process_annotations(self):

        qid_to_question = self.load_questions()
        doc_id_to_doc = self.load_documents()

        data_file = f'{self.root}/samples.jsonl'

        qas = defaultdict(list)
        with open(data_file) as infile:
            for line in infile:
                sample = json.loads(line) 
                qid = sample['qid']
                question = qid_to_question[qid].lower()

                docid = sample['docid']
                doc = doc_id_to_doc[docid]

                answers = [sample[f'answer{i}'] for i in range(1, 4) if sample[f'answer{i}'] ]
                if len(answers) > 0:
                    sample = {'text': doc, 'answers': answers, 'doc_id': docid, 'qid': qid}
                    qas[question].append(sample)
            
        visited = {}
        with open(f'{self.root}/annotations.jsonl', 'w') as outfile:
            for q, qa in qas.items():
                question = q
                qid = qa[0]['qid']
                if not qid in visited:
                    visited[qid] = True
                else:
                    continue
                
                docs = [{'doc_id': doc['doc_id'], 'title': '', 'text': doc['text'], 'answers': doc['answers']} for doc in qa]
                json.dump({'qid': qid, 'question': question, 'documents': docs}, outfile)
                outfile.write('\n')
        return 

    def process_passages(self, n=100):
        '''Split document text by n words'''
        ids, texts, titles = [], [], []
        count = 0

        with open(f'{self.root}/documents.jsonl') as infile:
            for j, line in enumerate(infile):
                ex = json.loads(line)

                if (j+1) % 1000000 == 0:
                    logger.info("Processed %s docs; total %s passages" % (j+1, count+1))

                title = ex['title']
                all_psgs = ex['text'].split(' ')
                pi = 0
                for i in range(0, len(all_psgs), n):
                    text = ' '.join(all_psgs[i:i + n])
                    ids.append('%s-%s' % (ex['doc_id'], pi))
                    texts.append(text)
                    titles.append(title)
                    pi+=1
                    count += 1

        logger.info(f'Total {j+1} documents processed; {count} passages processed.')

        with open(f'{self.root}/passages.jsonl', 'w') as outfile:
            for pid, title, text in zip(ids, titles, texts):
                json.dump({'pid': pid, 'title': title, 'text': text}, outfile)
                outfile.write('\n')


    def aggregate_answers(self):
        qrels = []
        q_tokens, ans_tokens, ans_num = [], [], []
        with open(f'{self.root}/annotations.jsonl') as infile:
            for line in infile:
                ex = json.loads(line)
                qid = ex['qid']
                question = ex['question']

                answers = []
                for doc in ex['documents']:
                    for ans in doc['answers']:
                        if ans not in answers:
                            answers.append(ans)
                            ans_tokens.append(len(ans.split(' ')))
                ans_num.append(len(answers))

                qrels.append({'qid': qid, 'question': question, 'answers': answers})
                q_tokens.append(len(question.split(' ')))
                

        logger.info(f'{len(qrels)} unique questions')
        logger.info(f'Average question tokens: {np.mean(q_tokens)}')
        logger.info(f'Average answer tokens: {np.mean(ans_tokens)}')
        logger.info(f'Average answers per question: {np.mean(ans_num)}')
        
        with open(f'{self.root}/qrels.jsonl', 'w') as outfile:
            for qrel in qrels:
                json.dump(qrel, outfile)
                outfile.write('\n')

