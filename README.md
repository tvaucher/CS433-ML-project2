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

### Create the environment

Please, also read the notes bellow if you don't have a GPU or a Mac

```shell
conda create -n cs433proj2 python=3.7 spacy pandas matplotlib
conda activate cs433proj2
conda install pytorch cudatoolkit=10.1 -c pytorch
pip install git+https://github.com/pytorch/text.git
```

**Notes on dependencies**

- [`pandas`]( https://pandas.pydata.org/ ):  Data structures and analysis tools. Used for data exploration in combination of `matplotlib` .
- [`spacy`]( https://spacy.io/ ): NLP framework. Used for tokenizing and during the preprocessing
- [`pytorch`]( https://pytorch.org/ ): Deep Learning framework. Used to create, train and exploit our deep learning models. If you're running on a laptop without a GPU, you need to change the `cudatoolkit=10.1` to `cpu_only`, or remove it completely on a Mac because they don't support CUDA at all. See [PyTorch Get Started]( https://pytorch.org/get-started/locally/ ) for more infos.
- [`torchtext `]( https://pytorch.org/text/index.html ): Allow to facilitate: Data loading and preprocessing, batches and feeding to a neural net as well as downloading and loading pretrained Embeddings (for us [Stanford NLP Twitter GloVe]( https://nlp.stanford.edu/projects/glove/ )). At the time of writing, `torchtext 0.4.0` has a [bug]( https://github.com/pytorch/text/pull/584 ) in one of its core features so we need to download it directly from GitHub in order to get the fixed version.

### Run the code

From the root folder of the project

```shell
python run.py
```

## Authors (team: Definitely not GRUs)

- Louis Amaudruz
- Andrej Janchevski
- Timoté Vaucher