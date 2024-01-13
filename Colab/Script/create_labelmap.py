import os
import xml.etree.ElementTree as ET
import time
from concurrent.futures import ThreadPoolExecutor

PIC_EXT = {'.png', '.jpg', '.jpeg', '.gif'}  # Use a set for faster membership tests


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

    # Check if the first two files have allowed extensions
    first_file_path = os.path.join(folder_path, files[0])
    second_file_path = os.path.join(folder_path, files[1])

    if not first_file_path.endswith(tuple(PIC_EXT)):
        print(f'{folder_path} does contain {first_file_path} wich is not a picture')
        return False

    if not second_file_path.lower().endswith('.xml'):
        print(f'{folder_path} does contain {second_file_path} wich is not a xml')
        return False

    return True


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


# Record the start time
start_time = time.time()

extract_path = 'images'

# Step 4: Identify subfolders within the extracted folder
folders_to_process = [subfolder for subfolder in os.listdir(extract_path) if
                      os.path.isdir(os.path.join(extract_path, subfolder))]

# Process files and XML
unique_classes = process_files_and_xml(folders_to_process)
print(unique_classes)

labelmap_path = '/tfrecord/labelmap.txt'

with open(labelmap_path, 'w') as f:
    for item_name in unique_classes:
        f.write(f'{item_name}\n')

# Record the end time
end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")
