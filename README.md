# CS-433 Project 2, 2019, Text Classification

## Authors (team: Definitely not GRUs)

- Louis Amaudruz
- Andrej Janchevski
- Timoté Vaucher

## Abstract

In this project, we take a look at the binary classification of tweets. We need to predict whether the original tweet contained a positive or a negative emoji. To this end, we first use state-of-the-art data preprocessing, identify task-specific important features and devise four models: a classic ML baseline, a GRU model using GloVe embeddings and two transfer-learning models based on ULMfit and BERT respectively. The best classifier we found is the BERT model which yields a 0.904 accuracy and F-1 score on the test set in the competition.

### For the reviewer

To run our final model for the evaluation, please proceed to the [BERT model](bert/README.md) README to get the setup and information. If you wish to consult other models, please proceed to their corresponding folders.

## Results

[Link](https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019/leaderboards) to the Competition leaderboard. Our team finished **2nd** out of **37** participating teams / indivduals.

| Model                             | Accuracy | F1-score |
| --------------------------------- | -------- | -------- |
| [Classic ML](classic_ml/)         | 0.770    | 0.783    |
| [GloVe + GRUs](gru-dl/)           | 0.881    | 0.883    |
| [ULMfit](ULMfit/)                 | 0.885    | 0.886    |
| [BERT](bert/) (bert-base-uncased) | 0.904    | 0.904    |

