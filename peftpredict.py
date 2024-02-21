import pandas as pd
from transformers import DistilBertTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from peft import AutoPeftModelForSequenceClassification

# Assuming AutoPeftModelForSequenceClassification is imported correctly for PEFT models
# from peft import AutoPeftModelForSequenceClassification

def predict_and_save(model_path, output_file, use_peft=False):
    # Initialize the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Dynamically load the model based on whether PEFT is used
    if use_peft:
        # Load the PEFT model from the specified path
        model = AutoPeftModelForSequenceClassification.from_pretrained(model_path)
    else:
        # Load the standard model from the specified path
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.eval()  # Set the model to evaluation mode
    
    # Load a random sample of entries from the IMDb dataset
    dataset = load_dataset("imdb", split='test').shuffle(seed=42).select(range(10))
    
    # Tokenize the texts
    inputs = tokenizer(dataset['text'], padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs).logits
    
    # Calculate probabilities and determine predictions
    probabilities = torch.softmax(outputs, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1).numpy()
    
    # Prepare the results
    results = []
    for i, (text, prediction) in enumerate(zip(dataset['text'], predicted_classes)):
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        results.append({"Sample": i+1, "Text": text, "Predicted Sentiment": sentiment})
    
    # Convert results to a pandas DataFrame
    df_results = pd.DataFrame(results)
    
    # Write the DataFrame to a CSV file
    df_results.to_csv(output_file, index=False)

# Predict and save results for the default (non-PEFT) model
predict_and_save("./default_model/", "default_model_predicted_sentiments.csv", use_peft=False)

# Predict and save results for the PEFT model
predict_and_save("./fine_tuned_model/", "peft_model_predicted_sentiments.csv", use_peft=True)
