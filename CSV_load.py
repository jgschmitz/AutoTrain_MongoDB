from datasets import load_dataset

# Load the CSV file
dataset_dict = load_dataset('csv', data_files='/Users/jeffery.schmitz/downloads/archive/RecipeNLG_dataset.csv')

# Access the 'train' split (or whatever split the dataset has)
dataset = dataset_dict['train']  # You may need to replace 'train' with the actual key, like 'train' or 'test'

# Select the first 1000 rows
small_dataset = dataset.select(range(1000))

# Push to Hugging Face Hub
small_dataset.push_to_hub("Schmitz005/recipe_nlg_dataset_sample")
