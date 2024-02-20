import pandas as pd
from transformers import DistilBertTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch

def predict_and_save(model_path, output_file):
    # Initialize the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load the model from the specified path
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode
    
    # Load a random sample of 5 entries from the IMDb dataset
    dataset = load_dataset("imdb", split='test').shuffle(seed=42).select(range(10))
    
    # Tokenize the texts
    inputs = tokenizer(dataset['text'], padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs).logits
    
    # Calculate probabilities and determine predictions
    probabilities = torch.softmax(outputs, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1).numpy()
    
    # Prepare the results for display without text truncation
    results = []
    for i, (text, prediction) in enumerate(zip(dataset['text'], predicted_classes)):
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        results.append({"Sample": i+1, "Text": text, "Predicted Sentiment": sentiment})  # Use full text
    
    # Convert results to a pandas DataFrame
    df_results = pd.DataFrame(results)
    
    # Write the DataFrame to a CSV file
    df_results.to_csv(output_file, index=False)

# Predict and save results for the default model
predict_and_save("./default_model/", "default_model_predicted_sentiments.csv")

# Predict and save results for the LoRA model continued
predict_and_save("./results_lora/checkpoint-2000/", "lora_model_predicted_sentiments.csv")