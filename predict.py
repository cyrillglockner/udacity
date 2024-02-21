import pandas as pd
from transformers import DistilBertTokenizer
from datasets import load_dataset
import torch
# Import the appropriate PEFT and LoRA classes
from peft import AutoPeftModelForSequenceClassification, LoraConfig, TaskType

def predict_and_save(model_path, output_file, use_lora=False):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    if use_lora:
        # Assume we have a function `load_lora_model` to handle LoRA-specific loading
        model = load_lora_model(model_path)
    else:
        # Standard loading for models without LoRA
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.eval()
    dataset = load_dataset("imdb", split='test').shuffle(seed=42).select(range(10))
    inputs = tokenizer(dataset['text'], padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs).logits
    
    probabilities = torch.softmax(outputs, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1).numpy()
    
    results = []
    for i, (text, prediction) in enumerate(zip(dataset['text'], predicted_classes)):
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        results.append({"Sample": i+1, "Text": text, "Predicted Sentiment": sentiment})
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)

def load_lora_model(model_path):
    
    target_modules = [
    "distilbert.transformer.layer.0.attention.out_lin",  # Attention output for the first layer
    "distilbert.transformer.layer.0.ffn.lin1",  # First linear layer of the FFN for the first layer
    "distilbert.transformer.layer.0.ffn.lin2",  # Second linear layer of the FFN for the first layer
    "distilbert.transformer.layer.1.attention.q_lin",  # Query in the second layer
    "distilbert.transformer.layer.1.attention.v_lin",  # Value in the second layer
    "distilbert.transformer.layer.1.attention.out_lin"  # Attention output for the second layer
    #"distilbert.transformer.layer.1.ffn.lin1",  # First linear layer of the FFN for the second layer
    #"distilbert.transformer.layer.1.ffn.lin2",  # Second linear layer of the FFN for the second layer
]

    lora_config = LoraConfig (
        task_type=TaskType.SEQ_CLS,  # Sequence classification task
        r=4,  # Adjusting rank for more capacity
        lora_alpha=2,  # Adjusting adaptation scale
        lora_dropout=0.1,  # Regularization via dropout
        target_modules=target_modules  # Targeting additional modules for LoRA adaptation
    )
    model = AutoPeftModelForSequenceClassification.from_pretrained(model_path, config=lora_config)
    
    return model

# Example usage
predict_and_save("default_model", "default_model_predictions.csv", use_lora=False)
predict_and_save("fine_tuned_model", "lora_model_predictions.csv", use_lora=True)
