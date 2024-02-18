from transformers import DistilBertTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import time
from sklearn.metrics import accuracy_score
from peft import LoraConfig, TaskType, get_peft_model  

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

test_dataset = load_dataset("imdb", split="test[:10%]")
test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
test_dataset = test_dataset.map(lambda e: tokenizer(e['text'], padding='max_length', truncation=True, max_length=512), batched=True)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

def evaluate_with_trainer(model_path, model_identifier, test_dataset, apply_lora=False):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    if apply_lora:
        # Define your LoRA configuration here, as per training script
        target_modules = [
            "distilbert.transformer.layer.0.attention.out_lin",
            "distilbert.transformer.layer.0.ffn.lin1",
            "distilbert.transformer.layer.0.ffn.lin2",
            "distilbert.transformer.layer.1.attention.q_lin",
            "distilbert.transformer.layer.1.attention.v_lin",
            "distilbert.transformer.layer.1.attention.out_lin",
            "distilbert.transformer.layer.1.ffn.lin1",
            "distilbert.transformer.layer.1.ffn.lin2",
        ]
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=4,
            lora_alpha=2,
            lora_dropout=0.1,
            target_modules=target_modules
        )
        model = get_peft_model(model, config)  # Apply LoRA to the model
    
    training_args = TrainingArguments(
        output_dir=f'./results_{model_identifier}',
        per_device_eval_batch_size=16,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )
    
    start_time = time.time()
    results = trainer.evaluate(test_dataset)
    inference_time = time.time() - start_time
    
    return results.get('eval_accuracy', None), inference_time

# Define clear model identifiers
baseline_identifier = "baseline"
lora_identifier = "lora"

# Evaluate baseline model without LoRA configurations
baseline_accuracy, baseline_time = evaluate_with_trainer("./default_model/", baseline_identifier, test_dataset, apply_lora=False)
# Evaluate LoRA model with LoRA configurations applied
lora_accuracy, lora_time = evaluate_with_trainer("./results_lora/checkpoint-2000/", lora_identifier, test_dataset, apply_lora=True)

print(f"Baseline Model Accuracy: {baseline_accuracy:.4f}, Inference Time: {baseline_time:.2f} seconds")
print(f"LoRA Model Accuracy: {lora_accuracy:.4f}, Inference Time: {lora_time:.2f} seconds")
