import pandas as pd

# Load the predictions CSV
results_df = pd.read_csv('predictions.csv')

# Extract true labels from filenames
results_df['true_label'] = results_df['filename'].str.extract(r'label_(\d+)', expand=False).astype(int)

# Calculate the number of correct predictions
correct_predictions = (results_df['true_label'] == results_df['prediction']).sum()

# Calculate accuracy
accuracy = correct_predictions / len(results_df)
print(f"Accuracy: {accuracy:.2%}")
