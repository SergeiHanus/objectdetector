import os
import shutil
from sklearn.model_selection import train_test_split

def organize_dataset():
    """Organize annotated data into train/val/test splits"""
    
    # Directories should already exist from setup
    for split in ['train', 'val', 'test']:
        os.makedirs(f'data/{split}/images', exist_ok=True)
        os.makedirs(f'data/{split}/labels', exist_ok=True)
    
    # Get all annotated images (those with corresponding .txt files)
    raw_images = []
    labels_dir = 'data/raw'  # LabelImg saves labels next to images
    
    for img_file in os.listdir('data/raw'):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            if os.path.exists(f'data/raw/{label_file}'):
                raw_images.append(img_file)
    
    print(f"Found {len(raw_images)} annotated images")
    
    if len(raw_images) < 10:
        print("⚠️  Warning: Very few annotated images. Consider collecting more data.")
    
    # Split data: 70% train, 20% val, 10% test
    train_imgs, temp_imgs = train_test_split(raw_images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)
    
    # Copy files to appropriate directories
    def copy_files(img_list, split_name):
        for img_file in img_list:
            # Copy image
            src_img = f'data/raw/{img_file}'
            dst_img = f'data/{split_name}/images/{img_file}'
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            src_label = f'data/raw/{label_file}'
            dst_label = f'data/{split_name}/labels/{label_file}'
            shutil.copy2(src_label, dst_label)
    
    copy_files(train_imgs, 'train')
    copy_files(val_imgs, 'val')
    copy_files(test_imgs, 'test')
    
    print(f"Data split: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    
    # Update dataset config with absolute path
    project_root = os.path.abspath('.')
    config_path = 'config/dataset.yaml'
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    content = content.replace('PLACEHOLDER_PATH', project_root + '/data')
    
    with open(config_path, 'w') as f:
        f.write(content)
    
    print(f"Updated dataset config: {config_path}")

if __name__ == "__main__":
    import sys
    
    # Verify we're running from the correct directory
    if not (os.path.exists('venv') and os.path.exists('data') and os.path.exists('src')):
        print("❌ Error: This script must be run from the project root directory")
        print("   Expected structure: venv/, data/, src/, config/ at same level")
        print("   Current directory:", os.getcwd())
        print("   Please run from: /data/code/image-detector/")
        print("   Usage: python src/data_preparation.py")
        sys.exit(1)
    
    organize_dataset()
