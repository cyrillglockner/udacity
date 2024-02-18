# Project Files Description

## `gptmodel.py`
Illustrates training and evaluating a distilibert model on the IMDb dataset, focusing on data loading, model configuration, training process, and performance metrics evaluation.

## `lora.py`
Demonstrates training a distilbert model with Low-Rank Adaptation (LoRA) on the IMDb dataset. It showcases data preprocessing, LoRA configuration, model training, and evaluation.

## `compare.py`
Script for evaluating and comparing the performance of baseline and LoRA fine-tuned models on the IMDb dataset. It calculates accuracy and inference times.

## `predict.py`
Generates sentiment predictions for input text samples.

## `requirements.txt`
Lists essential Python packages (torch, transformers, datasets, pandas, numpy) required for running the project. Use `pip install -r requirements.txt` to install dependencies.


# Known Issues

warnings about "Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']" 

Do not seem to influence the model performance


