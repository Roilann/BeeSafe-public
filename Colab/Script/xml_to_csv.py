# Created by Roilann with ChatGPT

import os
import xml.etree.ElementTree as ET
import pandas as pd


def extract_info_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text

    size_elem = root.find('size')
    width = int(size_elem.find('width').text)
    height = int(size_elem.find('height').text)

    objects_info = []
    for obj_elem in root.findall('object'):
        class_label = obj_elem.find('name').text
        bbox_elem = obj_elem.find('bndbox')
        xmin = int(bbox_elem.find('xmin').text)
        xmax = int(bbox_elem.find('xmax').text)
        ymin = int(bbox_elem.find('ymin').text)
        ymax = int(bbox_elem.find('ymax').text)

        obj_info = {
            'filename': filename,
            'width': width,
            'height': height,
            'class': class_label,
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax
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


# Print extracted information
def result_log():
    for obj_info in result:
        print(obj_info)


path = '/content/images/'
for folder in ['train', 'valid']:
    image_path = path + folder
    result = process_folder(image_path)
    
    # Create DataFrame from results
    df = pd.DataFrame(result)
    
    # Save DataFrame in a CSV file
    csv_filename = folder + '_labels.csv'
    df.to_csv(path + csv_filename, index=False)
    print(f"Successfully saved in {csv_filename}.")
