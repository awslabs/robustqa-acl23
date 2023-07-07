from argparse import ArgumentParser
from searchqa import SearchqaDataProcessor
from bioasq import BioASQDataProcessor
from fiqa import FiQADataProcessor
from lotte import LoTTE

DATAMAP = {
           'searchqa': SearchqaDataProcessor,
           'bioasq': BioASQDataProcessor,
           'fiqa': FiQADataProcessor,
           'lotte': LoTTE
          }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--split', default='test', type=str)
    args = parser.parse_args()

    # only LoTTE data has dev/test splits
    if args.data in ['lifestyle', 'recreation', 'technology', 'science', 'writing']:
        root = f'{args.data_dir}/{args.data}/{args.split}'
        data_processor = DATAMAP['lotte'](root)
    else:
        root = f'{args.data_dir}/{args.data}'
        data_processor = DATAMAP[args.data](root)

    data_processor.process_documents()
    data_processor.process_annotations()
    data_processor.process_passages()
    data_processor.aggregate_answers()