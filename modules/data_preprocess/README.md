# data_preprocess

## Datasets
### Dataset Struct
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
| Name           | Description                                                  | Language |
| -------------- | ------------------------------------------------------------ | -------- |
| People’s Daily | a collection of newswire article annotated with 3 balanced entity types | Chinese  |
| Boson_NER      | a set of online news annotations published by bosonNLP, which contains 6 entity types | Chinese  |
| Weibo_NER      | a collection of short blogs posted on Chinese social media Weibo with 8 extremely unbalanced entity types | Chinese  |
| OntoNotes-5.0  | a collection of broadcast news articles, which contains 18 entity types | Chinese  |
| CONLL2003      | a well known english dataset consists of Reuters news stories between August 1996 and August 1997, which contains 4 different entity types | English  |
| Ritter         | a english dataset consist of tweets annotated with 10 different entity types | English  |


### Basic Statistics
| Name                   | #S     | #T      | #E   | ASL   | ASE  | AEL  | %PT   | %AC   | %DAC  |
| ---------------------- | ------ | ------- | ---- | ----- | ---- | ---- | ----- | ----- | ----- |
| BosonNLP-train         | 27350  | 409830  | 6    | 14.98 | 0.67 | 3.93 | 17.7% | 41.8% | 14.7% |
| BosonNLP-test          | (6825) | (99616) | 6    | 14.59 | 0.67 | 3.87 | 17.8% | 41.8% | 14.8% |
| Weibo_NER-train        | 3664   | 85571   | 8    | 23.35 | 0.62 | 2.60 | 6.9%  | 33.6% | 14.8% |
| Weibo_NER-test         | 591    | 13810   | 8    | 23.36 | 0.66 | 2.60 | 7.3%  | 36.3% | 17.7% |
| OntoNotes5.0_NER-train | 13798  | 362508  | 18   | 26.27 | 1.91 | 3.14 | 22.8% | 72.5% | 48.0% |
| OntoNotes5.0_NER-test  | 1710   | 44790   | 18   | 26.19 | 1.99 | 3.07 | 23.4% | 75.4% | 51.5% |
| PeopleDaily-train      | 50658  | 2169879 | 3    | 42.83 | 1.47 | 3.23 | 11.1% | 58.3% | 35.8% |
| PeopleDaily-test       | 4620   | 172590  | 3    | 37.35 | 1.33 | 3.25 | 11.6% | 54.4% | 29.1% |
| CONLL2003-train        | 13862  | 203442  | 4    | 14.67 | 1.69 | 1.44 | 16.7% | 79.9% | 44.2% |
| CONLL2003-test         | 3235   | 51347   | 4    | 15.87 | 1.83 | 1.44 | 16.7% | 80.4% | 48.8% |
| Ritter-train           | 1955   | 37735   | 10   | 19.30 | 0.62 | 1.65 | 5.3%  | 38.1% | 15.3% |
| Ritter-test            | 438    | 8733    | 10   | 19.93 | 0.60 | 1.62 | 4.9%  | 39.2% | 15.5% 


