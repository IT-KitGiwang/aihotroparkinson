import os
import shutil
import random

# Define paths
dataset_dir = 'Dataset'
train_dir = 'Dataset/train'
test_dir = 'Dataset/test'

# Create train and test directories if they don't exist
os.makedirs(os.path.join(train_dir, 'Healthy'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'Parkinson'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'Healthy'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'Parkinson'), exist_ok=True)

# Function to split dataset for a class
def split_class(class_name):
    class_dir = os.path.join(dataset_dir, class_name)
    images = [f for f in os.listdir(class_dir) if f.endswith('.png')]
    random.shuffle(images)

    # Split 80/20
    split_index = int(0.8 * len(images))
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Copy to train
    for img in train_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, class_name, img))

    # Copy to test
    for img in test_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(test_dir, class_name, img))

    print(f'{class_name}: {len(train_images)} train, {len(test_images)} test')

# Split Healthy
split_class('Healthy')

# Split Parkinson
split_class('Parkinson')

print('Dataset split completed.')