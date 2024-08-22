import pandas as pd
import numpy as np
from PIL import Image
import os

# Load the CSV file
csv_file = 'fashion-mnist_test.csv'
data = pd.read_csv(csv_file)

# Directory to save images
output_dir = 'test_images'
os.makedirs(output_dir, exist_ok=True)

# Extract pixel data and labels
labels = data['label']
pixels = data.drop('label', axis=1).values

# Convert each row of pixel data to an image
for i, pixel_data in enumerate(pixels):
    image = pixel_data.reshape(28, 28).astype(np.uint8) 
    img = Image.fromarray(image) 
    label = labels[i]
    img.save(os.path.join(output_dir, f'image_{i}_label_{label}.png')) 

print(f"Saved {len(pixels)} images to {output_dir}")