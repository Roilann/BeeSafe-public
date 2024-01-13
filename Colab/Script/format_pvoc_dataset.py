# Created by Roilann with the help of ChatGPT

# How to use it:
# 1) place it in a folder with the zip file of the dataset
# 2) verify that there is no other zip file in this folder

# Result
# You should get a folder named "tfrecord" and 4 file inside "labelmap.pbtxt" "test.tfrecord" "train.tfrecord" "valid.tfrecord"

import os
import shutil
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

PIC_EXT = {'.png', '.jpg', '.jpeg', '.gif'}  # Use a set for faster membership tests


def contains_subfolder(folder_path):
    if os.path.isdir(folder_path):
        subfolders = [item for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]
        if subfolders:
            return True, subfolders
        else:
            return False, []
    else:
        raise ValueError(f"The provided path '{folder_path}' is not a valid directory.")


def create_folder(folder_path):
    # Check if the folder already exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def process_xml(xml_path, unique_object_names):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Iterate through each "object" in the XML
        for obj_elem in root.findall('.//object'):
            name_elem = obj_elem.find('name')

            if name_elem is not None:
                object_name = name_elem.text

                # Add the name to the set
                unique_object_names.add(object_name)

    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_path}: {e}")


def check_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        return False

    files = os.listdir(folder_path)

    # Check if there are at least two files in the folder
    if len(files) < 2:
        print(f' Less than 2 files in {folder_path}')
        return False

    # Sort files to ensure consistent order (e.g., alphabetical)
    files.sort()

    # Check if the first two files have allowed extensions
    first_file_path = os.path.join(folder_path, files[0])
    second_file_path = os.path.join(folder_path, files[1])

    if not first_file_path.endswith(tuple(PIC_EXT)):
        print(f'{folder_path} does contain {first_file_path} which is not a picture')
        return False

    if not second_file_path.lower().endswith('.xml'):
        print(f'{folder_path} does contain {second_file_path} which is not a xml')
        return False

    return True


def save_csv(data, csv_filename):
    df = pd.DataFrame(data)
    df.to_csv(csv_filename, index=False)
    print(f"Successfully saved in {csv_filename}.")


def process_files_and_xml(folders):
    unique_object_names = set()
    folder_check = 0

    for folder in folders:
        folder_path = os.path.join('images', folder)
        print(f'Checking {folder}')

        if not check_folder(folder_path):
            continue

        xml_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
                     filename.lower().endswith('.xml')]

        with ThreadPoolExecutor() as executor:
            executor.map(lambda xml_path: process_xml(xml_path, unique_object_names), xml_files)

        folder_check += 1

    if folder_check == len(folders):
        return True, list(unique_object_names)
    else:
        return False, []


def extract_info_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text

    size_elem = root.find('size')
    width = int(size_elem.find('width').text)
    height = int(size_elem.find('height').text)

    objects_info = []
    for obj_elem in root.findall('object'):
        bbox_elem = obj_elem.find('bndbox')
        obj_info = {
            'filename': filename,
            'width': width,
            'height': height,
            'class': obj_elem.find('name').text,
            'xmin': int(bbox_elem.find('xmin').text),
            'xmax': int(bbox_elem.find('xmax').text),
            'ymin': int(bbox_elem.find('ymin').text),
            'ymax': int(bbox_elem.find('ymax').text)
        }
        objects_info.append(obj_info)

    return objects_info


def process_folder(folder_path):
    all_objects_info = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            xml_file_path = os.path.join(folder_path, filename)
            objects_info = extract_info_from_xml(xml_file_path)
            all_objects_info.extend(objects_info)

    return all_objects_info


def process_folder_parallel(folder_path):
    all_objects_info = []
    xml_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.xml')]

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(extract_info_from_xml, xml_files))

    for result in results:
        all_objects_info.extend(result)

    return all_objects_info


def unzip_file(zip_path, extract_path):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


# Measure execution time
start_time = time.time()

# Step 1: Get the current script directory and create needed directories
script_directory = os.path.dirname(os.path.abspath(__file__))
# Folder that contains images
images_path = os.path.join(script_directory, 'images')
create_folder(images_path)
# Folder to work with
output_path = os.path.join(script_directory, 'tfrecord')
create_folder(output_path)

# Step 2: Identify the zip file(s) and extracted folder dynamically
zip_files = [file for file in os.listdir(script_directory) if file.endswith('.zip')]
nb_zip_files = len(zip_files)
if nb_zip_files != (1 and 3):
    raise ValueError(f"Error: There should be 1 zip file or 3 in the directory but there was {nb_zip_files}.")

# Step 3: Unzip data and place it in unzip
for zip_file in zip_files:
    extracted_folder = os.path.splitext(zip_file)[0]

    # Step 3.1: Unzip the identified zip file into the extracted folder
    zip_path = os.path.join(script_directory, zip_file)
    extract_path = os.path.join(script_directory, extracted_folder)
    print(f"Start {zip_file} extraction")
    unzip_file(zip_path, extract_path)
    print(f"{zip_file} is extracted")

    # Step 3.2: Identify subfolders within the extracted folder
    subfolders = [subfolder for subfolder in os.listdir(os.path.join(script_directory, extracted_folder)) if
                  os.path.isdir(os.path.join(script_directory, extracted_folder, subfolder))]

    result, subfolders = contains_subfolder(os.path.join(script_directory, extracted_folder))

    if result:
        # Step 3.3: Process subfolders (train, test, valid)
        print(f"Start copy of folders")
        for folder_name in subfolders:
            folder_path = os.path.join(extract_path, folder_name)
            folder_images_path = os.path.join(images_path, folder_name)

            shutil.move(folder_path, folder_images_path)

    # Step 3.4: Delete the extracted folder and its contents
    shutil.rmtree(extract_path)

# Step 4: Creates labelmap.txt
# Get classes from xml
result, unique_classes = process_files_and_xml(subfolders)
if result:
    print(f'classes found : {unique_classes}')
    # Print classes to labelmap
    labelmap_path = os.path.join(output_path + 'labelmap.txt')
    with open(labelmap_path, 'w') as f:
        for item_name in unique_classes:
            f.write(f'{item_name}\n')

# Step 8: xml_to_csv
print(f"Start csv conversion")
for folder in ['train', 'valid']:
    folder_path = os.path.join(images_path, folder)
    result = process_folder_parallel(folder_path)

    csv_filename = os.path.join(images_path, f"{folder}_labels.csv")
    save_csv(result, csv_filename)

# Step 9: tfrecord_conversion
print(f"Start tfrecord conversion")

# Time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time} seconds")

# Deletion
# shutil.rmtree(output_path)
# os.remove(os.path.join(script_directory, 'train_labels.csv'))
# os.remove(os.path.join(script_directory, 'valid_labels.csv'))

# Time
end_time_deletion = time.time()
elapsed_time_deletion = end_time_deletion - end_time
print(f"Deletion time: {elapsed_time_deletion} seconds")
complete_elapsed_time = end_time_deletion - start_time
print(f"Complete Execution time: {complete_elapsed_time} seconds")
