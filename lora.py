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
# Hint: Check the documentation at https://huggingface.co/transformers/v4.2.2/training.html
for param in model.parameters():
    param.requires_grad = True


# Preparing LoRA configuration
from transformers import AutoModelForSequenceClassification
from peft import LoraModel, LoraConfig, TaskType

target_modules = [
    "distilbert.transformer.layer.0.attention.out_lin",  # Attention output for the first layer
    "distilbert.transformer.layer.0.ffn.lin1",  # First linear layer of the FFN for the first layer
    "distilbert.transformer.layer.0.ffn.lin2",  # Second linear layer of the FFN for the first layer
    "distilbert.transformer.layer.1.attention.q_lin",  # Query in the second layer
    "distilbert.transformer.layer.1.attention.v_lin",  # Value in the second layer
    "distilbert.transformer.layer.1.attention.out_lin",  # Attention output for the second layer
    "distilbert.transformer.layer.1.ffn.lin1",  # First linear layer of the FFN for the second layer
    "distilbert.transformer.layer.1.ffn.lin2",  # Second linear layer of the FFN for the second layer
]

config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence classification task
    r=4,  # Adjusting rank for more capacity
    lora_alpha=2,  # Adjusting adaptation scale
    lora_dropout=0.1,  # Regularization via dropout
    target_modules=target_modules  # Targeting additional modules for LoRA adaptation
)

# Applying LoRA

from peft import get_peft_model
lora_model = get_peft_model(model, config)
print(lora_model)

# Check dataset size

print("Training set size:", len(tokenized_dataset["train"]))
print("Validation set size:", len(tokenized_dataset["test"]))

# Training the LoRA model
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# Transform datasets to torch format

class MyDataset(Dataset):
    def __init__(self, tokenized_dataset, split):
        self.tokenized_dataset = tokenized_dataset[split]
    
    def __len__(self):
        return len(self.tokenized_dataset['input_ids'])
    
    def __getitem__(self, idx):
        input_ids = self.tokenized_dataset['input_ids'][idx]
        attention_mask = self.tokenized_dataset['attention_mask'][idx]
        labels = self.tokenized_dataset['labels'][idx]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# Assign tokenized datasets
train_dataset = MyDataset(tokenized_dataset, 'train')
eval_dataset = MyDataset(tokenized_dataset, 'test')


# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results_lora/',
    evaluation_strategy='epoch',
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    #logging_level='info',
    learning_rate=2e-5
)

# Define the Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model

trainer.train()

# Evaluate the model
#
evaluation = trainer.evaluate()

# Print the evaluation metrics
print("Evaluation metrics:", evaluation)


model.save_pretrained("./fine_tuned_model/")

### In case you want to keep training ###
'''
## Load the LoRA-tuned model instead of initializing a new one
lora_model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model/")

# No need to reapply LoRA configuration here since the model already has it from the previous training session

# Define the Trainer with the previously trained LoRA model
training_args = TrainingArguments(
    output_dir='./results_lo_ra_continued/',  # Change to a new directory to save the continued training results
    evaluation_strategy='epoch',
    num_train_epochs=1,  # Adjust epochs depending on how much more you want to train
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs_continued',  # Change to a new directory for continued training logs
    logging_steps=10,
    learning_rate=2e-5  # Adjust the learning rate for continued training if desired
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,  # Assuming train_dataset is prepared as before
    eval_dataset=eval_dataset,  # Assuming eval_dataset is prepared as before
    compute_metrics=compute_metrics  # Assuming compute_metrics is defined as before
)

# Continue training
trainer.train()

# Evaluate the model after continued training
evaluation = trainer.evaluate()

# Print the evaluation metrics
print("Evaluation metrics:", evaluation)

# Save the further trained model
lora_model.save_pretrained("./fine_tuned_model_continued/")
'''