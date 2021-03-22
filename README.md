# AL-NER

In this project, we use pre-trained word2vec to implement word embedding, choose BiLstm as encoder and CRF as decoder.To evaluate the active learning strategies, we also implement several sample selection strategies based on uncertainty.

```
@misc{liu2020ltp,
title={LTP: A New Active Learning Strategy for CRF-Based Named Entity Recognition},
author={Mingyi Liu and Zhiying Tu and Tong Zhang and Tonghua Su and Zhongjie Wang},
year={2020},
eprint={2001.02524},
archivePrefix={arXiv},
primaryClass={cs.CL}
}
```

## Word Embedding

In this project, we use a 300d word embedding pre-trained on the Chinese Wikipedia corpus for the Chinese datasets, and a 100d glove word
embedding pre-trained on the English Wikipedia corpus for the English.
You can get them from the download link below, then you need to convert these files to .pkl files.
+ [merge_sgns_bigram_char300.txt](https://pan.baidu.com/s/14JP1gD7hcmsWdSpTvA3vKA?errmsg=Auth+Login+Sucess&errno=0&ssnerror=0&)
+ [glove.6B.100d.txt](http://212.129.155.247/embedding/glove.6B.100d.zip)

## BiLstm-CRF

BiLstm-CRF has been widely used in named entity recognition on several typical datasets.

## Sample selection strategies

In this project, the following selection strategies are implemented.

+ RANDOM: RandomStrategy
+ LC: LeastConfidenceStrategy
+ NLC: NormalizedLeastConfidenceStrategy
+ LTP: LeastTokenProbabilityStrategy
+ MTP: MinimumTokenProbabilityStrategy
+ MTE: MaximumTokenEntropyStrategy
+ LONG: LongStrategy
+ TE: TokenEntropyStrategy

## Prerequisites

* python 3.6
* pytorch 1.5.1
* numpy 1.19.1
* sklearn 0.23.1
* seqeval
* colorama


## Datasets

We have experimented and evaluate the active learning strategies mentioned above on four Chinese datasets and two english datasets.We get these datasets from the dounload link below, then carry out some data preprocessing operations on these files, such as dividing the extra long sentences through ','.

### Dataset Struct

You can find some sample files which contain part of the datasets under the directory of datasets.In this project, we store datasets in the following structure.

```
datasets
    |
    | --- dataset1
                | --- train.txt
                | --- test.txt
                | --- tags.txt
    | --- dataset2
                | --- train.txt
                | --- test.txt
                | --- tags.txt
    | --- ....
```

### Basic Description For Each Dataset

| Name                                                         | Description                                                  | Language |
| ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| [People’s Daily](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily) | a collection of newswire article annotated with 3 balanced entity types | Chinese  |
| [Boson_NER](https://bosonnlp.com/resources/BosonNLP_NER_6C.zip) | a set of online news annotations published by bosonNLP, which contains 6 entity types | Chinese  |
| [Weibo_NER](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/Weibo) | a collection of short blogs posted on Chinese social media Weibo with 8 extremely unbalanced entity types | Chinese  |
| [OntoNotes-5.0](https://catalog.ldc.upenn.edu/LDC2013T19)    | a collection of broadcast news articles, which contains 18 entity types | Chinese  |
| [CONLL2003](https://github.com/patverga/torch-ner-nlp-from-scratch/tree/master/data/conll2003/) | a well known english dataset consists of Reuters news stories between August 1996 and August 1997, which contains 4 different entity types | English  |
| [Ritter](https://github.com/aritter/twitter_nlp/blob/master/data/annotated/ner.txt) | a english dataset consist of tweets annotated with 10 different entity types | English  |

### Basic Statistics

| Name                   | #S     | #T      | #E   | ASL   | ASE  | AEL  | %PT   | %AC   | %DAC  |
| ---------------------- | ------ | ------- | ---- | ----- | ---- | ---- | ----- | ----- | ----- |
| BosonNLP-train         | 27350  | 409830  | 6    | 14.98 | 0.67 | 3.93 | 17.7% | 41.8% | 14.7% |
| BosonNLP-test          | 6825   | 99616   | 6    | 14.59 | 0.67 | 3.87 | 17.8% | 41.8% | 14.8% |
| Weibo_NER-train        | 3664   | 85571   | 8    | 23.35 | 0.62 | 2.60 | 6.9%  | 33.6% | 14.8% |
| Weibo_NER-test         | 591    | 13810   | 8    | 23.36 | 0.66 | 2.60 | 7.3%  | 36.3% | 17.7% |
| OntoNotes5.0_NER-train | 13798  | 362508  | 18   | 26.27 | 1.91 | 3.14 | 22.8% | 72.5% | 48.0% |
| OntoNotes5.0_NER-test  | 1710   | 44790   | 18   | 26.19 | 1.99 | 3.07 | 23.4% | 75.4% | 51.5% |
| PeopleDaily-train      | 50658  | 2169879 | 3    | 42.83 | 1.47 | 3.23 | 11.1% | 58.3% | 35.8% |
| PeopleDaily-test       | 4620   | 172590  | 3    | 37.35 | 1.33 | 3.25 | 11.6% | 54.4% | 29.1% |
| CONLL2003-train        | 13862  | 203442  | 4    | 14.67 | 1.69 | 1.44 | 16.7% | 79.9% | 44.2% |
| CONLL2003-test         | 3235   | 51347   | 4    | 15.87 | 1.83 | 1.44 | 16.7% | 80.4% | 48.8% |
| Ritter-train           | 1955   | 37735   | 10   | 19.30 | 0.62 | 1.65 | 5.3%  | 38.1% | 15.3% |
| Ritter-test            | 438    | 8733    | 10   | 19.93 | 0.60 | 1.62 | 4.9%  | 39.2% | 15.5% |


## Usage

1. Modify the configuration file as required (al-ner-demo/config_files/bilstm-crf-al.config)

```
[LOGGER]
logdir_prefix=../logger # please make sure that this directory exists

[WORDEMBEDDING]
method=WORD2VEC 

[MODELTRAIN]
method=BiLSTMCRF

[WORD2VEC]
all_word_embedding_path=../embedding/merge_sgns_bigram_char300.pkl
choose_fraction=0.01
courpus_file=../datasets/BosonNLP_NER_6C/
courpus_name=BosonNLP_NER_6C
embedding_dim=300
entity_type=6
max_seq_len=64
tags_file=../datasets/BosonNLP_NER_6C/tags.txt

[BiLSTMCRF]
batch_size=64
device=cuda:0
embedding_dim=300
hidden_dim=200
num_rnn_layers=1
num_epoch=25
learning_rate=1e-3
model_path_prefix=../model/word2vec_bilstm_crf_ltp # please make sure that ../model exists

[ENTITYLEVELF1]
average=micro
digits=2
return_report=False

[ActiveStrategy]
strategy=LTP #other options:RANDOM,LC,NLC,MNLP,MTP,MTE,LONG,TE
stop_echo=25
query_batch_fraction=0.02
```

According to the above configuration file, log file will be saved under the directory below.
`AL-NER/logger/BosonNLP_NER_6C/WORD2VEC_BiLSTMCRF_LTP/`

2. Type the command line and try to run it

```
cd al-ner-demo/pipelines/
python -u Word2VecBiLSTMCRFALPipeline.py -c ../config_files/bilstm-crf-al.config -t 1-2 --project 00001
```

