# Detect mask, use yolo and keras 

## Development Process
1. Use official darknet to train (YOLOv4: https://github.com/AlexeyAB/darknet/)
2. Convert to Tensorflow model & save as .pb files  (Officail TF model: https://github.com/hunglc007/tensorflow-yolov4-tflite)
3. Use Keras load model and use it

## Requirements: Tensorflow 2.5:arrow_up:  , opencv
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


# How do I train my model?
