import json
import logging
from processor import DataProcessor
from utils import flatten_answers
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BioASQDataProcessor(DataProcessor):

    def process_documents(self):
        '''Documents taken from the MeSH 2021 release'''
        
        with open(f'{self.root}/allMeSH_2021.json', encoding='windows-1252') as infile:
            data = json.load(infile)['articles']
            logger.info("Total: %s" % len(data))

        with open(f'{self.root}/documents.jsonl', 'w') as outfile:
            for i, doc in enumerate(data):
                json.dump({'title': doc['title'], 'text': doc['abstractText'], 'doc_id': doc['pmid'], 'metadata':{'year': doc['year']}}, outfile)
                outfile.write('\n')
                if (i+1) % 1000000 == 0:
                    logger.info(i+1)

        logger.info("Done!")

    def process_annotations(self):
        samples = []
        for i in range(2, 10):
            qid = 0
            for j in range(1, 6):
                with open(f'{self.root}/test/{i}B{j}_golden.json') as infile:
                    data = json.load(infile)
                    for ques in data['questions']:
                        if ques['type'] in ['factoid', 'list']:
                            question = ques['body'].lower()

                            # answers are already aggregated. Pair them with documents to be consistent with other data's format
                            ans = ques['exact_answer']
                            if type(ans) == list:
                                ans = flatten_answers(ans)
                            assert all([type(x) is str for x in ans])
                            
                            pos_cxts = []
                            for doc in ques['documents']:
                                doc_id = doc.split('/')[-1]
                                pos_cxts.append({'doc_id': doc_id, 'answers': ans})

                            samples.append({"qid": 'test%s' % qid, "question": question, "documents": pos_cxts})
                            qid += 1

        logger.info(len(samples))

        with open(f'{self.root}/annotations.jsonl', 'w') as outfile:
            for sample in samples:
                json.dump(sample, outfile)
                outfile.write('\n')
