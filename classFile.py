from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset('Schmitz005/kaggle-recipe-categorized')

# Convert the 'source' column to ClassLabel
dataset = dataset.class_encode_column('source')

# Verify the conversion
print(dataset['train'].features)

# Optionally push the dataset back to Hugging Face if you want to store the fixed version
dataset.push_to_hub('Schmitz005/kaggle-recipe-categorized-fixed')
