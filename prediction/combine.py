import os
import shutil

# Paths to original folders
train_dir = 'C:\\Users\\Medha Agarwal\\Desktop\\GANs\\brain-tumor-mri-dataset\\Testing'
test_dir = 'C:\\Users\\Medha Agarwal\\Desktop\\GANs\\brain-tumor-mri-dataset\\Testing'

# Create combined dataset folder
combined_dir = 'path_to_combined_dataset'
os.makedirs(combined_dir, exist_ok=True)

# Get class names from the training folder (assuming both have same classes)
classes = os.listdir(train_dir)

# Create class subfolders in the combined directory
for cls in classes:
    os.makedirs(os.path.join(combined_dir, cls), exist_ok=True)

# Function to copy files from source to destination
def copy_files(src_folder, dest_folder):
    for cls in classes:
        src_class_dir = os.path.join(src_folder, cls)
        dest_class_dir = os.path.join(dest_folder, cls)
        
        for filename in os.listdir(src_class_dir):
            shutil.copy(os.path.join(src_class_dir, filename), os.path.join(dest_class_dir, filename))

# Copy files from training and testing folders
copy_files(train_dir, combined_dir)
copy_files(test_dir, combined_dir)

print(f"Images from training and testing sets have been merged into {combined_dir}")
