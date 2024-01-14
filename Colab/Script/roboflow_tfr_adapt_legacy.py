# Mainly created by Roilann with the help of ChatGPT

# How to use it:
# 1) place it in a folder with the zip file of the dataset
# 2) verify that there is no other zip file in this folder

# Result
# You should get a folder named "tfrecord" and 4 file inside
# "labelmap.pbtxt" "test.tfrecord" "train.tfrecord" "valid.tfrecord"

import os
import shutil
import re
from zipfile import ZipFile


def contains_subfolder(folder_path):
    if os.path.isdir(folder_path):
        subfolders = [item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]
        if subfolders:
            return True, subfolders
        else:
            return False, []
    else:
        raise ValueError(f"The provided path '{folder_path}' is not a valid directory.")


def parse_labelmap(labelmap_content):
    items = re.findall(r'item\s*{([^}]+)}', labelmap_content)
    result = []

    for item in items:
        # Extract name using regex
        item_name = re.search(r'name:\s*["\']([^"\']+)["\']', item).group(1)
        result.append(item_name)

    return result


def create_folder(folder_path):
    # Check if the folder already exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def unzip_file(zip_path, extract_path):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


# Step 1: Get the current script directory
script_directory = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_directory, 'tfrecord')
create_folder(output_path)
labelmap_txt_path = os.path.join(output_path, 'labelmap.txt')
labelmap_pbtxt_path = os.path.join(output_path, 'labelmap.pbtxt')

# Step 2: Identify the zip file(s) and extracted folder dynamically
zip_files = [file for file in os.listdir(script_directory) if file.endswith('.zip')]
nb_zip_files = len(zip_files)
if nb_zip_files != 1 and nb_zip_files != 3:
    raise ValueError(f"Error: There should be 1 zip file or 3 in the directory but there was {nb_zip_files}.")

for i, zip_file in zip_files:
    extracted_folder = os.path.splitext(zip_file)[0]

    # Step 3: Unzip the identified zip file into the extracted folder
    zip_path = os.path.join(script_directory, zip_file)
    extract_path = os.path.join(script_directory, extracted_folder)
    unzip_file(zip_path, extract_path)

    # Step 4: Identify subfolders within the extracted folder
    result, subfolders = contains_subfolder(extract_path)

    # Step 6: Process subfolders (train, test, valid)
    if result:
        for folder_name in subfolders:
            folder_path = os.path.join(extract_path, folder_name)
            tf_file = [file for file in os.listdir(folder_path) if file.endswith('.tfrecord')]

            if len(tf_file) > 1:
                raise ValueError(f"Error: More than one .tfrecord file found in folder {folder_name}")
            if len(tf_file) == 0:
                raise ValueError(f"Error: No .tfrecord file found in folder {folder_name}")

            tf_file_path = os.path.join(folder_path, tf_file[0])
            tf_file_output_path = os.path.join(output_path, f"{folder_name}.tfrecord")

            os.rename(tf_file_path, tf_file_output_path)

    # Step 7: Get the labelmap
    if i == 0:
        first_subfolder_path = os.path.join(extract_path, subfolders[0])
        labelmap_file = [file for file in os.listdir(first_subfolder_path) if file.endswith('.pbtxt')]
        labelmap_path = os.path.join(first_subfolder_path, labelmap_file[0])
        os.rename(labelmap_path, labelmap_pbtxt_path)

    # Step 9: Delete the extracted folder and its contents
    shutil.rmtree(extract_path)

# Step 8: Take labelmap.pbtxt to create labelmap.txt tf standard
with open(labelmap_pbtxt_path, 'r') as file:
    labelmap_content = file.read()

parsed_data = parse_labelmap(labelmap_content)

with open(labelmap_txt_path, 'w') as f:
    for item_name in parsed_data:
        f.write(f'{item_name}\n')
