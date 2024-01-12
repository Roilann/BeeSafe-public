# Mainly created by Roilann with the help of ChatGPT
# Creation of labelmap part comes from the source below
# GitHub user datitran: https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py

# How to use it:
# 1) place it in a folder with the zip file of the dataset
# 2) verify that there is no other zip file in this folder

# Result
# You should get a folder named "tfrecord" and 4 file inside "labelmap.pbtxt" "test.tfrecord" "train.tfrecord" "valid.tfrecord"

import os
import shutil
import re
from zipfile import ZipFile


def parse_labelmap(labelmap_content):
    items = re.findall(r'item\s*{([^}]+)}', labelmap_content)
    result = []

    for item in items:
        # Extract id, name, and display_name using regex
        item_info = {'id': int(re.search(r'id:\s*(\d+)', item).group(1)),
                     'name': re.search(r'name:\s*["\']([^"\']+)["\']', item).group(1)}

        result.append((item_info['id'], item_info['name']))

    return result


def unzip_file(zip_path, extract_path):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


def main():
    # Step 1: Get the current script directory
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Step 2: Identify the only zip file and extracted folder dynamically
    zip_files = [file for file in os.listdir(script_directory) if file.endswith('.zip')]
    if len(zip_files) != 1:
        print("Error: There should be exactly one zip file in the current directory.")
        return

    zip_file = zip_files[0]
    extracted_folder = os.path.splitext(zip_file)[0]

    # Step 3: Unzip the identified zip file into the extracted folder
    zip_path = os.path.join(script_directory, zip_file)
    extract_path = os.path.join(script_directory, extracted_folder)
    unzip_file(zip_path, extract_path)

    # Step 4: Identify subfolders within the extracted folder
    subfolders = [subfolder for subfolder in os.listdir(extract_path) if
                  os.path.isdir(os.path.join(extract_path, subfolder))]

    # pour + tard
    # file_count = len([f for f in os.listdir('path') if os.path.isfile(os.path.join('path', f))])
    # os.makedirs('folder_path')
    # os.rename('old', 'new')

    # Step 5: Create the output folder (tfrecord)
    output_path = os.path.join(script_directory, 'tfrecord')
    os.makedirs(output_path)

    # Step 6: Process subfolders (train, test, valid)
    for folder_name in subfolders:
        folder_path = os.path.join(extract_path, folder_name)
        tf_file = [file for file in os.listdir(folder_path) if file.endswith('.tfrecord')]
        tf_file_path = os.path.join(folder_path, tf_file[0])
        tf_file_output_path = os.path.join(output_path, f"{folder_name}.tfrecord")

        os.rename(tf_file_path, tf_file_output_path)

    # Step 7: Get the labelmap
    first_subfolder_path = os.path.join(extract_path, subfolders[0])
    labelmap_file = [file for file in os.listdir(folder_path) if file.endswith('.pbtxt')]
    labelmap_path = os.path.join(first_subfolder_path, labelmap_file[0])
    labelmap_output_path = os.path.join(output_path, 'labelmap.pbtxt')

    # Step 8: Adapt Roboflow labelmap.pbtxt to tf standard
    with open(labelmap_path, 'r') as file:
        labelmap_content = file.read()

    parsed_data = parse_labelmap(labelmap_content)

    # datitran part
    with open(labelmap_output_path, 'w') as f:
        for item_id, item_name in parsed_data:
            f.write(f'item {{\n  id: {item_id}\n  name: \'{item_name}\'\n}}\n\n')

    # Step 9: Delete the extracted folder and its contents
    shutil.rmtree(extract_path)


if __name__ == "__main__":
    main()
