# Scripts for Colab
These scripts are supposed to work in Colab.
They are made so no arguments are passed

## roboflow_tfr_adapt.py
**Author**: *Roilann* + *ChatGPT*\
**Credits**: *[datitran](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py)*

**Explanation** : Format the files in a better way to train and test your AI in Colab.\
It also creates the labelmap.txt to visualize results\
<u>Input:</u> 1 ZIP file (with **train** / **valid** / **test** inside)\
<u>Output:</u> (3 tfrecord (**train** / **valid** / **test**) + 1 labelmap.pbtxt + 1 labelmap.txt) => inside a folder `tfrecord`

## tfrecord_generator.py
**Author**: *Roilann* + *ChatGPT*\
**Credits**: *[datitran](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py)* &
*[EdjeElectronics](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/util_scripts/create_tfrecord.py)*

**Explanation** : In case of PASCAL_VOC data, it generates tfrecord before the training configuration\
<u>Input:</u> 2 folder (**train** / **valid**) with PASCAL_VOC compiled data + 2 csv (**train** / **valid**) + labelmap.txt\
<u>Output:</u> (2 tfrecord (**train** / **valid**) + 1 labelmap.pbtxt) => inside a folder `tfrecord`

## tfrecord_to_png.py
**Author**: *Roilann* + *ChatGPT*

**Explanation** : Gets the **test** tfrecord and extract it to get the pictures as png\
Allows to visualize the result if files uploaded as tfrecord\
<u>Input:</u> 1 tfrecord (**test**)\
<u>Output:</u> X pictures inside `tfrecord` inside `images`

## xml_to_csv.py
**Author**: *Roilann* + *ChatGPT*\
**Credits**: *[datitran](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py)* &
*[EdjeElectronics](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/util_scripts/create_tfrecord.py)*

**Explanation** : Gets pascal_voc \
<u>Input:</u> 2 folder (**train** / **valid**) with PASCAL_VOC compiled data\
<u>Output:</u> 2 csv (**train** / **valid**)