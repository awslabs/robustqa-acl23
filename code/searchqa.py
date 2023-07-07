import glob
import json
import logging
from processor import DataProcessor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchqaDataProcessor(DataProcessor):
    
    def process_documents(self):
        '''Documents are pool across all splits'''

        doc_id = 0
        with open(f'{self.root}/documents.jsonl', 'w') as outfile:
            for split in ['train', 'val', 'test']:
                for filename in glob.glob(f'{self.root}/{split}/*.json'):
                    with open(filename) as infile:
                        data = json.load(infile)

                        for result in data['search_results']:
                            title = result['title']
                            text = result['snippet']
                            if not text or not title:
                                continue
                            else:
                                json.dump({'doc_id': doc_id, 'title': title, 'text': text, 'meta_data': {}}, outfile)
                                outfile.write('\n')
                                doc_id += 1

            logger.info('Total %s documents processed' % doc_id)

    def process_annotations(self):
        '''For QA we only need test split for evaluation'''

        def doc_mapping():
            # create document to docid mapping
            doc_mapping = {}
            with open(f'{self.root}/documents.jsonl') as infile:
                for line in infile:
                    ex = json.loads(line)
                    text_key = ex['title'] + ' ' + ex['text']
                    doc_mapping[text_key] = ex['doc_id']

            return doc_mapping
        
        doc_lookup = doc_mapping()
        
        samples = []
        count, no_answer, qid = 0, 0, 0
        for filename in glob.glob(f'{self.root}/test/*.json'):
            with open(filename) as infile:
                data = json.load(infile)
                question = data['question']
                answer = data['answer']

                if not answer:
                    no_answer += 1
                    continue
                
                pos_num, pos_cxts = 0, []
                for result in data['search_results']:
                    title = result['title']
                    doc = result['snippet']

                    if not doc or not title:
                        continue
                    else:
                        if answer.lower() in doc.lower():
                            pos_num += 1
                            lookup_key = title + ' ' + doc
                            pos_cxts.append({'title': title, 'text': doc, 'doc_id': doc_lookup[lookup_key], 'answers': [answer]})

                if pos_num == 0:
                    no_answer += 1
                else:
                    sample = {
                              "qid": 'test%s' % qid,
                              "question": question.lower(),
                              "documents": pos_cxts
                              }
                    qid += 1
                    samples.append(sample)
                count += 1

        logger.info("%d\t%d\t%.4f" % (count, count - no_answer, no_answer / count))
        logger.info(len(samples))

        with open(f'{self.root}/annotations.jsonl', 'w') as outfile:
            for sample in samples:
                json.dump(sample, outfile)
                outfile.write('\n')