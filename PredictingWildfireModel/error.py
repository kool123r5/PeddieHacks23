import os
from PIL import Image

def check_for_corrupted_images(directory):
    corrupted_images = []

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            try:
                #some guys scammed us kinda
                img = Image.open(file_path)
                img.load()
            except Exception as e:
                corrupted_images.append(file_path)
                print(f"Corrupted image: {file_path} - {e}")

    return corrupted_images

def remove_files(file_paths):
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

image_root_directory = '/Users/kushalb/Downloads/fireData/test/wildfire'

corrupted_images = check_for_corrupted_images(image_root_directory)

if not corrupted_images:
    print("No corrupted images found.")
else:
    print(f"Total corrupted images: {len(corrupted_images)}")
    print("Removing corrupted images...")
    remove_files(corrupted_images)
    print("Removal complete.")
