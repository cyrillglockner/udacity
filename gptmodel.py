# Load IMDB Dataset and split into train and test data

from datasets import load_dataset, DatasetDict

# Load the IMDb dataset's train split and then split it further into train and test subsets
split_dataset = load_dataset("imdb", split="train").train_test_split(test_size=0.2, shuffle=True, seed=23)

# Create a DatasetDict to manage the train/test splits easily
dataset = DatasetDict({
    'train': split_dataset['train'].shuffle(seed=42).select(range(int(len(split_dataset['train']) * 0.2))),
    'test': split_dataset['test'].shuffle(seed=42).select(range(int(len(split_dataset['test']) * 0.2)))
})

splits = ["train", "test"]

# Rename 'label' column to 'labels'
dataset = dataset.rename_column("label", "labels")

# Verify the column has been renamed
print("Columns after renaming:", dataset.column_names)

# Print the dataset information
print(dataset)

# Inspect the first example. Is this a positive or negative review?
dataset["train"][1]

# Tokenize dataset

from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenized_dataset = {}
for split in splits:
    tokenized_dataset[split] = dataset[split].map(
        lambda x: tokenizer(x["text"], truncation=True,padding=True), batched=True
    )
    
# Inspect the available columns in the dataset
tokenized_dataset["train"]
print(tokenized_dataset)

# Setup metrics
from datasets import load_metric
import numpy as np

accuracy_metric = load_metric("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return accuracy_metric.compute(predictions=preds, references=labels)
    

# Prepare model

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "Positive", 1: "Negative"},
    label2id={"Positive": 0, "Negative": 1},
)

# Unfreeze all the model parameters.
# Check the documentation at https://huggingface.co/transformers/v4.2.2/training.html
for param in model.parameters():
    param.requires_grad = True

# Train and evaluate model performance
import torch
from torch.utils.data import DataLoader

# Import additional transformer classes
from transformers import Trainer, TrainingArguments

# Prepare the inputs for the model
input_ids = tokenized_dataset['train']['input_ids']
attention_mask = tokenized_dataset['train']['attention_mask']
labels = tokenized_dataset['train']['labels']

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
evaluation = trainer.evaluate()

# Print the evaluation metrics
print("Evaluation metrics:", evaluation)

model.save_pretrained("./default_model/")

