# Scripts for Colab
These scripts are supposed to work in Colab.
They are made so no arguments are passed

## create_labelmap.py
**Author**: *Roilann* + *ChatGPT*

**Explanation** : Get the 3 pascal_voc folders (**train** / **valid** / **test**) and run through them to get the different classes saved in a labelmap.txt\
<u>Input:</u> 3 files (**train** / **valid** / **test**) within the folder `images` in pascal_voc format\
<u>Output:</u> 1 labelmap.txt inside a folder `tfrecord`

## format_dataset.py
**Author**: *Roilann* + *ChatGPT*\
**Credits**: *[datitran](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py)* &
*[EdjeElectronics](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/util_scripts/create_tfrecord.py)*

**Explanation** : Is a combination of all the scripts in this folder\
<u>Input:</u> 1 zip file (with **train** / **valid** / **test** inside) or 3 zip files (**train** / **valid** / **test**) in PVOC or TFrecord format\
<u>Output:</u> (3 tfrecord (**train** / **valid** / **test**) + 1 labelmap.pbtxt + 1 labelmap.txt) => inside a folder `tfrecord`

## format_pvoc_dataset.py
**Author**: *Roilann* + *ChatGPT*\
**Credits**: *[datitran](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py)* &
*[EdjeElectronics](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/util_scripts/create_tfrecord.py)*

**Explanation** : Get 3 zip files (**train** / **valid** / **test**) or 1 zip (with **train** / **valid** / **test** inside) in PVOC format to prep data to train.\
<u>Input:</u> 1 zip file (with **train** / **valid** / **test** inside) or 3 zip files (**train** / **valid** / **test**) in PVOC format\
<u>Output:</u> (3 tfrecord (**train** / **valid** / **test**) + 1 labelmap.pbtxt + 1 labelmap.txt) => inside a folder `tfrecord`

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