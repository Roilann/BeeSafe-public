import os
import tensorflow as tf
from PIL import Image

def _parse_function(proto):
    # Define your feature names and types here
    keys_to_features = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string, default_value=''),  # Set a default value
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }

    # Parse the input tf.Example proto using the dictionary above
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # Handle the case where 'image/source_id' is not present
    if parsed_features['image/source_id'] == '':
        parsed_features['image/source_id'] = tf.constant('', dtype=tf.string)

    return parsed_features

def load_tfrecord_dataset(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset

def extract_images(dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i, example in enumerate(dataset):
        image_encoded = example['image/encoded'].numpy()

        # Save images in PNG format with simple convention
        image_path = os.path.join(output_dir, f"test_output_{i}.png")
        with open(image_path, 'wb') as f:
            f.write(image_encoded)

        #print(f"Image {i+1} extracted to: {image_path}")

# Example usage
tfrecord_path = '/content/tfrecord/test.tfrecord'
output_directory = '/content/images/test/'
dataset = load_tfrecord_dataset(tfrecord_path)
extract_images(dataset, output_directory)
print(f"Images Successfully Extracted from tfrecord file")