import requests
import base64
import os
import pandas as pd

# Directory containing images to process
image_dir = '/Users/achshahrm/Documents/IU International/Sem 04/Model to Production/Project/Images'

# Prepare the list of base64-encoded images
image_list = []
filenames = []
for filename in os.listdir(image_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        with open(os.path.join(image_dir, filename), 'rb') as img_file:
            img_str = base64.b64encode(img_file.read()).decode('utf-8')
            image_list.append(img_str)
            filenames.append(filename)

# Send the request to the API
response = requests.post('http://localhost:5001/process_batch', json={'images': image_list})

# Check if the request was successful
if response.status_code == 200:
    try:
        predictions = response.json().get('predictions', [])
    except ValueError:
        print("Failed to decode JSON response")
        predictions = []
else:
    print(f"Request failed with status code {response.status_code}")
    predictions = []

# Save the predictions to a CSV file if available
if predictions:
    results_df = pd.DataFrame({
        'filename': filenames,
        'prediction': predictions
    })
    results_df.to_csv('predictions.csv', index=False)
else:
    print("No predictions were made.")