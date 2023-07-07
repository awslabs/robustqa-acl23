# RobustQA-ACL23-Data

This repo describes the details of the _[RobustQA (ACL'23 Findings)](https://www.amazon.science/publications/robustqa-benchmarking-the-robustness-of-domain-adaptation-for-open-domain-question-answering)_ benchmark, which consists of datasets in 8 domains.

| Domain     | Dataset  | Description | Adapted/Annotated? |
| :--------- | -------- | ----------- |     :---:          |
| Web Search | SearchQA | Jeopardy! QA based on Google search engine | Adapted |
| Biomedical | BioASQ   | Open-domain QA based on PubMed documents   | Adapted |
| Finance    | FiQA     | Financial QA based on microblogs, reports, news  | Annotated |
| Lifestyle  | LoTTE    | QA regrading lifestyle based on original IR data in search and forum | Annotated |
| Recreation | LoTTE    | QA regarding recreation based on original IR data in search and forum | Annotated |
| Technology | LoTTE    | QA regarding technology based on original IR data in search and forum | Annotated |
| Science    | LoTTE    | QA regarding science based on original IR data in search and forum | Annotated |
| Writing    | LoTTE    | QA regarding writing based on original IR data in search and forum | Annotated |
----- 


## Disclaimers
We've included the links to the license for each of the raw datasets. We only distribute some of the RobustQA's datasets in a specific format, but we do not vouch for their quality or fairness, or claim that you have license to use the dataset. It remains the user's responsibility to determine whether you as a user have permission to use the dataset under the dataset's license and to cite the right owner of the dataset.


## Citation

```
@Inproceedings{Han2023,
 author = {Rujun Han and Peng Qi and Yuhao Zhang and Lan Liu and Juliette Burger and William Wang and Zhiheng Huang and Bing Xiang and Dan Roth},
 title = {RobustQA: Benchmarking the robustness of domain adaptation for open-domain question answering},
 year = {2023},
 url = {https://www.amazon.science/publications/robustqa-benchmarking-the-robustness-of-domain-adaptation-for-open-domain-question-answering},
 booktitle = {ACL Findings 2023},
}

```

## Raw Data & Annotations
Due to data license and legal constraints, we could only provide partial final data, and our new annotations without raw data. You can find them in `data/.` All files in this folder are tracked by _[Git LFS](https://git-lfs.com/)_.


For the rest of the data, we provide instructions to download raw data, and process them into uniform data format for RobustQA. In general, after data processing, you can expect to have the following data and field,

- `documents.jsonl`: original document pool. Data fields are,
    - `doc_id`: document id
    - `title`: document title
    - `text`: document text
    - `meta_data`: optional
- `annotations.jsonl`: extractive QA annotations in the original document. Data fields are,
    - `qid`: question id
    - `question`: question text
    - `documents`: 
        - `answers`: answer span annotated in the document
        - `doc_id`: same as above
        - `title`: same as above
        - `text`: same as above
- `passages.jsonl`: split document texts in `documents.jsonl` by 100 words (based on white space). Data fields are the same as `document.jsonl` except for
    - `pid = doc_id-k` where k is the k-th split of a document (0-based)
- `qrel.jsonl`: aggregate all answers per question
    - `qid`:  question id
    - `question`: original question
    - `answers`: aggregated answers across different documents.

Passage file `passages.jsonl` and aggregated QA file `qrel.jsonl` are needed for the experiments in the paper. 

### FiQA
- License: There is no data license specfied https://sites.google.com/view/fiqa/home. However, due to Amazon legal requirements, we only deep `doc_id` and `qid` in the published annotation files.
- Download the raw corpus `FiQA_train_doc_final.tsv` and question file `FiQA_train_question_final.tsv` into `data/fiqa` from https://drive.google.com/file/d/1BlWaV-qVPfpGyJoWQJU9bXQgWCATgxEP/view.
- To replicate `documents.jsonl` and `annotations.jsonl`, run `python code/process_raw.py --dataset fiqa`


### LoTTE
- Download raw data here: https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz into `data/lotte`. 
- Annotations: there is no data license specfied https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md. However, due to Amazon legal requirements, we only keep `doc_id` and `qid` in the published annotation files.
- To replicate `documents.jsonl` and `annotations.jsonl`, run `python code/process_raw.py --dataset {lifestyle|recreation|technology|science|writing} --split {test|dev}`


### BioASQ
We only provide detailed data reproduction instruction and code below to avoid any potential issues per the following license. So, you will have to acquire the raw data on your own and run the following data processing code.
- License: https://creativecommons.org/licenses/by/2.5/
- Register an account here: http://bioasq.org/
- download document collecction `allMeSH_2021.zip` and `unzip allMeSH_2021.zip`
- download the test annotations `{2-9}B{1-5}_golden.json`
- move both documents and annotations to `data/bioasq/`
- `python code/process_raw.py --dataset bioasq`


### SearchQA
We only provide detailed data reproduction instruction and code below to avoid any potential issues per the following license. So, you will have to acquire the raw data on your own and run the following data processing code.
- License: https://github.com/nyu-dl/dl4ir-searchQA/blob/master/LICENSE
- Raw data - `{train|val|test}.zip` can be download from here: https://drive.google.com/drive/u/2/folders/1kBkQGooNyG0h8waaOJpgdGtOnlb1S649
- `mkdir -p data/searchqa/{train|val|test}`
- `mv {train|val|test}.zip data/searchqa/{train|val|test}`
- `unzip data/searchqa/{train|val|test}.zip`
- `python code/process_raw.py --dataset searchqa`


## Experiment - Passage Retrieval

### DPR
Following the instruction here:https://github.com/facebookresearch/DPR to install the DPR package, download NQ data and the trained models. By default,
- Passages are saved here: `DPR/downloads/data/` and qa annotations are saved here `DPR/downloads/data/retriever/qas/`. 
- Model checkpoints are saved here: `DPR/downloads/checkpoint/retriever/`
    - we use `single-adv-hn` model for retrieval experiments.
```
HOME=~/robustqa
OUTDIR=~/DPR/downloads/data/

# options: searchqa bioasq fiqa lifestyle recreation science technology writing
dataset=fiqa
cd ${OUTDIR}
mkdir ${dataset}_split

cd $HOME
python code/convert_to_dpr.py --data ${dataset} --output_dir ${OUTDIR}
```
Then follow the instruction on the same web-page to generate embeddings and retrieve passages.


### BM25 + CE
Refer to this instruction for details of installing BEIR package: https://github.com/beir-cellar/beir.
```
HOME=~/robustqa
# options: searchqa bioasq fiqa lifestyle recreation science technology writing
dataset=fiqa

cd $HOME
pip install beir
python code/convert_to_beir.py --${dataset}
```
Setup BM25,
```
wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.9.2-linux-x86_64.tar.gz
wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-oss-7.9.2-linux-x86_64.tar.gz.sha512
tar -xzf elasticsearch-oss-7.9.2-linux-x86_64.tar.gz
sudo chown -R daemon:daemon elasticsearch-7.9.2/
shasum -a 512 -c elasticsearch-oss-7.9.2-linux-x86_64.tar.gz.sha512
sudo -H -u daemon elasticsearch-7.9.2/bin/elasticsearch
```
Run models `python code/run_beir_models.py --data ${dataset} --model bm25 --reindex`


### ColBERTv2
Detailed instruction of running ColBERTv2 can be found here: https://github.com/stanford-futuredata/ColBERT. We do not repeat. After setting up the ColBERTv2 directories and environment, 
- move scripts to the ColBERT folder
- convert RobustQA data into ColBERT format
```
COLBERT=~/ColBERT
HOME=~/robustqa

cp colbert_scripts/* $COLBERT/

# options: searchqa bioasq fiqa lifestyle recreation science technology writing
dataset=fiqa

python code/convert_to_colbert.py --data ${dataset} --output_dir $COLBERT/data
```

Running ColBERTv2 consists of four steps,
1. Download ColBERTv2 checkpoint to `$COLBERT/downloads/colbertv2.0'`
2. Indexing passages 
```
python colbert_scripts/run_colbert.py \
    --dataroot ${COLBERT}/data/ \
    --dataset  ${dataset}
    --model $COLBERT/downloads/colbertv2.0 \
    --index
```
3. Search top passages
```
python colbert_scripts/run_colbert.py \
    --dataroot ${COLBERT}/data/ \
    --dataset  ${dataset}
    --model $COLBERT/downloads/colbertv2.0 \
    --search
```
This step will save a `*ranking.tsv` file into `$COLBERT/experiments`. Locate this file's path (`ranking_file_path`)
4. Compute performance and save retrieval results
```
python colbert_scripts/run_colbert.py \
    --dataroot ${COLBERT}/data/ \
    --dataset  ${dataset}
    --model $COLBERT/downloads/colbertv2.0 \
    --eval \
    --ranking_file ${ranking_file_path} 
```
This step will save a retrieved passage file `{dataset}_from_colbert_{split}.json` under `${COLBERT}/output/`. This file is in the same format of the retrieved passage file from DPR above, and can be directory used as the input to the extractive QA model.



## Experiment - Question Answering

### DPR's Extractive QA
To run the extractive QA model inference, download the best QA model checkpoint from https://github.com/facebookresearch/DPR. Using the following script,
```
python train_extractive_reader.py \
  prediction_results_file={path to a file to write the results to} \
  eval_top_docs=[10,20,40,50,80,100] \
  dev_files={path to the retriever results file to evaluate} \
  model_file= {path to the reader checkpoint} \
  train.dev_batch_size=80 \
  passages_per_question_predict=100 \
  encoder.sequence_length=350
```
Since we use ColBERTv2 as the default retriever for the paper, `dev_files` needs to be set to the path of the `{dataset}_from_colbert_{split}.json` files from ColBERTv2. See details above.


### Atlas
Atlas model doesn't not require ColBERTv2 to provide retrieved passages since it has its own dense retriever. Following the intruction here: https://github.com/facebookresearch/atlas to set up the project repo and install environment. Models checkpoints we experimented in the paper are `atlas-xxl_nq` and `atlas-base_nq`.

Convert RobustQA data into Atlas format,
```
ATLAS=~/atlas
HOME=~/robustqa

# options: searchqa bioasq fiqa lifestyle recreation science technology writing
dataset=fiqa

python code/convert_to_atlas.py --data ${dataset} --output_dir $ATLAS/data
```

Run inference,
```
cd $ATLAS
export NGPU=8
model_size=base
n_cxt=40
split=test

python -m torch.distributed.launch --nproc_per_node=8  evaluate.py \
    --name run_atlas_nq_${model_size}_${n_cxt}_${dataset} \
    --generation_max_length 16 \
    --gold_score_mode "pdist" \
    --precision bf16 \
    --per_gpu_embedder_batch_size 128 \
    --reader_model_type google/t5-${model_size}-lm-adapt \
    --text_maxlength 200 \
    --model_path $ATLAS/models/atlas_nq/${model_size} \
    --eval_data $ATLAS/data/${dataset}-${split}.jsonl \
    --per_gpu_batch_size 1 \
    --n_context ${n_cxt} --retriever_n_context ${n_cxt} \
    --main_port -1 \
    --index_mode "flat"  \
    --task "qa" \
    --passages  $ATLAS/data/${dataset}-passages.jsonl \
```

