import os
import xml.etree.ElementTree as ET
import time
from concurrent.futures import ThreadPoolExecutor

PIC_EXT = {'.png', '.jpg'}  # Use a set for faster membership tests


def process_xml(xml_path, unique_names):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Iterate through each "object" in the XML
        for obj_elem in root.findall('.//object'):
            name_elem = obj_elem.find('name')

            if name_elem is not None:
                object_name = name_elem.text

                # Add the name to the set
                unique_names.add(object_name)

    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_path}: {e}")


def process_files_and_xml(folders):
    # List to store unique object names
    unique_names = set()
    folder_check = 0

    for folder in folders:
        # Get the list of files in the specified folder
        folder_path = os.path.join('images', folder)
        files = os.listdir(folder_path)
        print(f'checking {folder}')

        # Check if there are at least two files in the folder
        if len(files) < 2:
            print(f' Less than 2 file in {folder}')
            continue

        # Check if the first two files have allowed extensions
        first_file_path = os.path.join(folder_path, files[0])
        second_file_path = os.path.join(folder_path, files[1])

        if not first_file_path.lower().endswith(tuple(PIC_EXT)):
            print(f'{folder} does not contain picture')
            continue

        if not second_file_path.lower().endswith('.xml'):
            print(f'{folder} does not contain xml')
            continue

        xml_files = [os.path.join(folder_path, filename) for filename in files if filename.lower().endswith('.xml')]

        # Use ThreadPoolExecutor to parallelize the XML processing
        # Useful when a lot of files to check
        with ThreadPoolExecutor() as executor:
            executor.map(lambda xml_path: process_xml(xml_path, unique_names), xml_files)

        folder_check += 1

    if folder_check == len(folders_path):
        return True, list(unique_names)
    else:
        return False, []


# Record the start time
start_time = time.time()

folders = ["train", "test", "valid"]

# Process files and XML
result, classes = process_files_and_xml(folders)
print(result)
print(classes)

labelmap_path = '/tfrecord/labelmap.txt'

with open(labelmap_path, 'w') as f:
    for item_name in classes:
        f.write(f'{item_name}\n')

# Record the end time
end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")
