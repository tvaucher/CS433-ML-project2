# CS-433 Project 2, 2019, Text Classification

## File structure

- `BERT_Full.ipynb`: Notebook containg the entire pipeline. Tested with Nvidia T4 and P100, approximate run time ~8h w/ P100.
- `helpers.py`: Helpers methods to set the seed, get the GPU, create batches and the submission file
- `preprocessing.py`: All the preprocessing methods for this project, adapted from gru-dl. 
- `run.py`: Contains the methods to load the test set and model, infer the prediction and output the submission file. See usage below. Approximate runtime on Nvidia P100 ~20 sec, Geforce MX150 (laptop GPU) ~4 min, on CPU ~15 min.

## Reproducibility

### Prerequisite
1. [Anaconda](https://www.anaconda.com/distribution/) with support for python 3.7 and [git](https://git-scm.com/downloads) are both installed

2. If you want to run just `run.py`, that the encoded data and model ([zip archive](https://drive.google.com/file/d/1QI8aiZpfazXo3BrFFrL9RmoLa4gIbmNO/view?usp=sharing), [tar ball](https://drive.google.com/file/d/1Jx6DpGfHrcHQq9MjySnYVYj7OMEVVSW8/view?usp=sharing)) is downloaded and extracted like (the file structure should already be in the compressed data):
------------

    ├── data
    │   ├── test_encoded_BERT.pkl.gz
    │   ├── train_encoded_BERT.pkl.gz
    ├── model
    │   ├── BERT
    │   |   ├── config.json
    │   |   ├── pytorch_model.bin

--------

3. Otherwise, if you want to run from strach, download the [official data](https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019/dataset_files) and extract it under `data`

### Create the environment

Please, also read the notes bellow if you don't have a GPU or a Mac

```shell
conda create -n twitter_bert python=3.7 pandas tqdm scikit-learn
conda activate twitter_bert
conda install pytorch=1.3 cudatoolkit=10.1 -c pytorch
pip install spacy==2.2 transformers==2.2.2
python -m spacy download en_core_web_sm
```

**Notes on dependencies**

- [`pandas`]( https://pandas.pydata.org/ ):  Data structures and analysis tools. Used for data exploration in combination of `matplotlib` .
- [`tqdm`](https://tqdm.github.io/): fancy progress bars
- [`spacy`]( https://spacy.io/ ): NLP framework. Used for tokenizing and during the preprocessing
- [`pytorch`]( https://pytorch.org/ ): Deep Learning framework. Used to exploit our deep learning model. If you're running on a laptop without a GPU, you need to change the `cudatoolkit=10.1` to `cpu_only`, or remove it completely on a Mac because they don't support CUDA at all. See [PyTorch Get Started]( https://pytorch.org/get-started/locally/ ) for more infos.
- [`Transformers`]( https://huggingface.co/transformers ): Transformers library, contain all recent models. Used for DistilBERT (during the testing of the library) and BERT (used for the final model)

### Run the code

From the root folder of the project

```shell
python run.py -f data/test_encoded_BERT.pkl.gz -o data/submission.csv -m model/BERT

# help
usage: run.py [-h] -f FILE -o OUT -m MODEL [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Path to test encoded file (.pkl.gz)
  -o OUT, --out OUT     Path to submission file (.csv)
  -m MODEL, --model MODEL
                        Path to model folder
  --batch_size BATCH_SIZE
                        Batch size (2GB GPU==10, else 25), default=10
```

If you want to run the entire project, please use `BERT_full.ipynb` on a Google Cloud w/ GPU instance or similar 

## Authors (team: Definitely not GRUs)

- Louis Amaudruz
- Andrej Janchevski
- Timoté Vaucher