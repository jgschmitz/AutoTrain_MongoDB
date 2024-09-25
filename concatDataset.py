from datasets import load_dataset

# Load the concatenated dataset
dataset = load_dataset('Schmitz005/kaggle-recipe-categorized-chunk-1')['train']

# Check unique values in the 'NER' column
print(dataset.unique('NER'))
