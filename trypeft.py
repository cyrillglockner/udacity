from peft import AutoPeftModelForSequenceClassification

# Assuming your PEFT model is saved in "./fine_tuned_model/"

peft_model = AutoPeftModelForSequenceClassification.from_pretrained("fine_tuned_model")

from transformers import AutoTokenizer
import torch

# Load the tokenizer corresponding to your base model (e.g., distilbert-base-uncased)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Example sequence
sequences = ["The quick brown fox jumps over the lazy dog", "An apple a day keeps the doctor away"]

# Tokenize sequences
inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# Perform classification
with torch.no_grad():
    outputs = peft_model(**inputs)

# Interpret logits or scores
logits = outputs.logits
predicted_labels = torch.argmax(logits, dim=1)

# Print predicted labels
print(predicted_labels)
