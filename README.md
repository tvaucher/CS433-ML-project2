# CS-433 Project 2, 2019, Text Classification
## Abstract

ToDo : Write a cool abstract

## Brief Overview

Lorem ipsum dolor sit amet 

## File structure

- foo.py
- bar.py
- baz.py

## Reproducibility

### Prerequisite
1. [Anaconda](https://www.anaconda.com/distribution/) with support for python 3.7 and [git](https://git-scm.com/downloads) are both installed

2. That the [data](https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019/dataset_files) is downloaded and extracted like :
------------

    ├── data
    │   ├── train_neg_full.txt
    │   ├── train_pos_full.txt
    │   ├── test_data.txt

--------

3. If you don't want to run the preprocessing part (about ~15 min) we suggest that you download our [preprocessed training dataset]() and put it in the datafolder

### Create the environment

Please, also read the notes bellow if you don't have a GPU or a Mac

```shell
conda create -n twitter python=3.7 pandas matplotlib tqdm
conda activate twitter
conda install pytorch=1.3 cudatoolkit=10.1 -c pytorch
pip install spacy==2.2 git+https://github.com/pytorch/text.git
python -m spacy download en_core_web_sm
```

**Notes on dependencies**

- [`pandas`]( https://pandas.pydata.org/ ):  Data structures and analysis tools. Used for data exploration in combination of `matplotlib` .
- [`tqdm`](https://tqdm.github.io/): fancy progress bars
- [`spacy`]( https://spacy.io/ ): NLP framework. Used for tokenizing and during the preprocessing
- [`pytorch`]( https://pytorch.org/ ): Deep Learning framework. Used to create, train and exploit our deep learning models. If you're running on a laptop without a GPU, you need to change the `cudatoolkit=10.1` to `cpu_only`, or remove it completely on a Mac because they don't support CUDA at all. See [PyTorch Get Started]( https://pytorch.org/get-started/locally/ ) for more infos.
- [`torchtext `]( https://pytorch.org/text/index.html ): Allow to facilitate: Data loading and preprocessing, batches and feeding to a neural net as well as downloading and loading pretrained Embeddings (for us [Stanford NLP Twitter GloVe]( https://nlp.stanford.edu/projects/glove/ )). At the time of writing, `torchtext 0.4.0` has a [bug]( https://github.com/pytorch/text/pull/584 ) in one of its core features so we use the latest version on github

### Run the code

From the root folder of the project

In order to perform the preprocessing:
```shell
python preprocessing.py -p data/train_pos_full.txt -n data/train_neg_full.txt -o data/train_full.tsv

python preprocessing.py -h # For the Help
```

In order to completly train the dataset:
```shell
# For example
python train.py -f data/train_full.tsv -g .vector_cache/ -c model/chkpt -n 5 --dim 200

# For the help
train.py [-h] -f FILE -g GLOVE -c CHECKPOINT [-n EPOCHS] [--seed SEED]
                [--dim {25,50,100,200}]

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Path to preprocessed training file (.tsv)
  -g GLOVE, --glove GLOVE
                        Path to the folder containing the pretrained GloVe
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Path to checkpoint prefix
  -n EPOCHS, --epochs EPOCHS
                        Number of epoch to train for default=10
  --seed SEED           Set the seed, default=1
  --dim {25,50,100,200}
                        Set the dimension of the embedding [25, 50, 100, 200],
                        default=200
```

In order to output the submission:
```shell
python run.py -f data/test_data.txt -o data/submission.csv -m model/model.pth -v model/vocabulary.pth
```

## Authors (team: Definitely not GRUs)

- Louis Amaudruz
- Andrej Janchevski
- Timoté Vaucher