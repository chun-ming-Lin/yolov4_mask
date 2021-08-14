
# Detect mask, use yolov4 and keras 

## Development Process
- 1. [Use official darknet to train](#train_model) (YOLOv4: https://github.com/AlexeyAB/darknet/)
- 2. [Convert to Tensorflow model & save as .pb files](#convert_model)  (Officail TF model: https://github.com/hunglc007/tensorflow-yolov4-tflite)
- 3. Use Keras load model and use it

## Requirements: Tensorflow 2.5:arrow_up:  , opencv 4.5.3
Optional: freetype

## How to use my python scripts
Download [My .h5 file](https://drive.google.com/file/d/16jJf6fI0iV-8I4oKKPJv9Nyowxsdb0WD/view?usp=sharing)

### Run python file on command line
```
# detect with your webcam
python detect_webcam.py

# detect video and save it
python detect_video.py

# if you have freetype module
detect_chinese.py
```
#### If you want build opencv with freetype module, check [this](https://github.com/BabaGodPikin/Build-OpenCv-for-Python-with-Extra-Modules-Windows-10)


# How do I train my model?
Download [dataset](https://drive.google.com/file/d/1qe0_oHBZGu6xDTgT-sLs3Qm87KsJteWu/view?usp=sharing)

- <a name="train_model"></a> I train my model on colab, this is my [Code](https://github.com/chun-ming-Lin/yolov4_mask/blob/master/colab/YOLOv4_mask.ipynb).  <br>
- <a name="convert_model"></a> Convert to tensorflow, this is my [Code](https://github.com/chun-ming-Lin/yolov4_mask/blob/master/colab/yolo_convert_keras_h5.ipynb)

