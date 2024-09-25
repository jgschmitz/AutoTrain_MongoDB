# 🤗 AutoTrain Hugging Face -> MongoDB
Repo to accompany Cisco Webinar on 09/25/2024

First install requirements.txt
```
pip install -r requirements.txt
```

then you can chunk your data  - season to taste
```
from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset('Schmitz005/kaggle-recipe')

# Total number of rows
total_rows = len(dataset['train'])

# Define chunk size (e.g., 100,000 rows at a time)
chunk_size = 100000

# Split the dataset into smaller chunks and process them
for i in range(0, total_rows, chunk_size):
    # Select a chunk of data
    chunk = dataset['train'].select(range(i, min(i + chunk_size, total_rows)))
    
    print(f"Processing chunk {i // chunk_size + 1}")
    
    # Process 'source' column in the chunk
    chunk = chunk.class_encode_column('source')
    
    # Optionally push each chunk to Hugging Face, or save it locally
    chunk.push_to_hub(f"Schmitz005/kaggle-recipe-categorized-chunk-{i // chunk_size + 1}")
```
You can then run this on your existing chunks
```
import subprocess
from huggingface_hub import login

# Log into Hugging Face with your write token
login(token="PUT YOUR HF TOKEN HERE")

# Run the AutoTrain text-generation command using subprocess
command = [
    "autotrain", "seq2seq",  # Task for sequence-to-sequence text generation
    "--train",  # Train the model
    "--data-path", "Schmitz005/kaggle-recipe-categorized-chunk-1",  # Use one of the chunks
    "--model", "t5-small",  # Using a small model like T5 for text generation
    "--text-column", "title",  # Input column (recipe title)
    "--target-column", "NER",  # Output column (list of ingredients)
    "--max-seq-length", "128",  # Maximum input/output length
    "--epochs", "3",  # Number of training epochs
    "--batch-size", "8",  # Batch size
]
```
Once this completes you should have a saved_model directory should have these files
```
config.json
model.safetensors
special_tokens_map.json
tokenizer_config.json
vocab.txt
```
You now have the model you can further tune and work with it as much as you like!
