# Project Files Description

## `gptmodel.py`
Training and evaluating a distilibert model on the IMDb dataset.

## `lora.py`
Taining a distilbert model with Low-Rank Adaptation (LoRA) on the IMDb dataset.

## `compare.py`
Script for evaluating and comparing the performance of baseline and LoRA fine-tuned models on the IMDb dataset.

## `predict.py`
Generates sentiment predictions for input text samples and stores results in two csv files.

## `requirements.txt`
Essential Python packages (torch, transformers, datasets, pandas, numpy) required for running the project. 
Use `pip install -r requirements.txt` to install dependencies.


# Known Issues

warnings about "Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']" 

Do not seem to influence the model performance

# Steps to use the repo

## Execute in order
pip install -r requirements.txt

python gptmodel.py 

python lora.py

python conmpare.py

python predict.py 



