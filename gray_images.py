import os
import cv2
import shutil


def convert_images_to_grayscale(src_folder, dst_folder, image_extensions=['.jpg', '.jpeg', '.png', '.bmp']):
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                # Construct the full file path
                src_file_path = os.path.join(root, file)

                # Compute the relative path to maintain folder structure
                rel_path = os.path.relpath(src_file_path, src_folder)
                dst_file_path = os.path.join(dst_folder, rel_path)

                # Create the destination directory if it does not exist
                os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

                # Load the image
                image = cv2.imread(src_file_path)

                # Convert the image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Save the grayscale image
                cv2.imwrite(dst_file_path, gray_image)

                print(f"Converted and saved: {dst_file_path}")


# Define the source and destination folders
src_folder = 'D:\PhD\Images\Adaptation_Dataset_Sample'
dst_folder = 'D:\PhD\Images\Grayscale'

# Create the destination folder if it doesn't exist
os.makedirs(dst_folder, exist_ok=True)

# Convert images
convert_images_to_grayscale(src_folder, dst_folder)