import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths to the image directories
image_dir_1 = '/Users/laksharajjha/Desktop/model/data/HAM10000_images_part_1'
image_dir_2 = '/Users/laksharajjha/Desktop/model/data/HAM10000_images_part_2'

# Load the metadata CSV file
metadata = pd.read_csv('/Users/laksharajjha/Desktop/model/HAM10000_metadata.csv')

# Function to load images from the directories
def load_data(metadata, img_dirs, img_size=(128, 128)):
    images = []
    labels = []
    
    for _, row in metadata.iterrows():
        img_id = row['image_id']
        label = row['dx']  # Use this for the classification label

        # Search for the image in the provided directories
        img_path = None
        for img_dir in img_dirs:
            if os.path.exists(os.path.join(img_dir, img_id + '.jpg')):
                img_path = os.path.join(img_dir, img_id + '.jpg')
                break

        # If an image is found, process it
        if img_path:
            try:
                image = Image.open(img_path).resize(img_size)
                images.append(np.array(image) / 255.0)  # Normalize pixel values
                labels.append(1 if label == 'mel' else 0)  # Binary classification (melanoma vs. others)
            except Exception as e:
                print(f"Error processing image {img_id}: {e}")

    return np.array(images), np.array(labels)

# Load data from both image directories
img_dirs = [image_dir_1, image_dir_2]
X, y = load_data(metadata, img_dirs)

# Print the number of images loaded
print(f"Total images loaded: {X.shape[0]}")

# Split the data into training and test sets and save to .npy files
if X.size > 0 and y.size > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    print("Files saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
else:
    print("No data to save. Please check the directories and metadata.")
