# Script to craft TFRecord files from train and valid dataset folders
# Created by GitHub user datitran: https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py
# Inspired by Github user EdjeElectronics: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/util_scripts/create_tfrecord.py
# Updated by Github user Roilann:

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd

from tensorflow.python.framework.versions import VERSION

if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple


def generate_tfrecord(csv_input_path, labelmap_path, image_dir_path, output_tfrecord_path):
    def split(df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

    def create_tf_example(group, path):
        with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        labels = []
        with open(labelmap_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(int(labels.index(row['class']) + 1))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    # Load and prepare data
    writer = tf.python_io.TFRecordWriter(output_tfrecord_path)
    path = os.path.join(os.getcwd(), image_dir_path)
    examples = pd.read_csv(csv_input_path)

    # Craft TFRecord files (.tfrecord)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_tfrecord_path)
    print('Successfully created the TFRecords: {}'.format(output_path))

    # Craft labelmap file (.pbtxt)
    path_to_labelpbtxt = os.path.join(os.getcwd(), 'labelmap.pbtxt')

    with open(labelmap_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    with open(path_to_labelpbtxt, 'w') as f:
        for i, label in enumerate(labels, start=1):
            f.write(f'item {{\n  id: {i}\n  name: \'{label}\'\n}}\n\n')


if __name__ == '__main__':
    script_directory = os.path.dirname(os.path.abspath(__file__))

    train_csv_path = os.path.join(script_directory, 'train_labels.csv')
    valid_csv_path = os.path.join(script_directory, 'valid_labels.csv')

    label_path = os.path.join(script_directory, 'labelmap.txt')

    images_train_path = os.path.join(script_directory, 'images/train')
    images_valid_path = os.path.join(script_directory, 'images/valid')

    output_train_path = os.path.join(script_directory, 'train.tfrecord')
    output_valid_path = os.path.join(script_directory, 'valid.tfrecord')

    # Example usage:
    generate_tfrecord(train_csv_path,
                      label_path,
                      images_train_path,
                      output_train_path)

    generate_tfrecord(valid_csv_path,
                      label_path,
                      images_valid_path,
                      output_valid_path)