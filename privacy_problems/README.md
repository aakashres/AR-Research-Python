# Code for face detection and recognition

This folder consists of files that will be used for face detection and recognition. The folder *training data* will hold all images that you capture for training purpose. 
 1. main.py
    The main python script file that will handle all the actions for this project.

 2. detectors.py
    This file contains various algorithm for face detection. If any algorithm needs to be implemented regarding face detection, Please add a function by looking the code structure.

 3. consts.py
    This file is for all the static variables value.

 4. face.py
    This file contains a class that represents the face with all necessary information regarding detection and recognition.

 5. utilities.py
    All the utilities function that is going to be reused is placed here.


# Usage
1. Command for help
```
python .\main.py --help

usage: main.py [-h] [-c CONFIDENCE] [-d DEVICE] [-n NAME] [-t DETECTOR_TYPE] [-s SAVE_FACE]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIDENCE, --confidence CONFIDENCE
                        Confidence Interval
  -d DEVICE, --device DEVICE
                        Input stream for video
  -n NAME, --name NAME  Name for the face. Necessary is save_face option is True
  -t DETECTOR_TYPE, --detector_type DETECTOR_TYPE
                        Detector Type ssd,haarcascade,ssd_andy
  -s SAVE_FACE, --save_face SAVE_FACE
                        Save faces for training. Any input value will be True. Leave empty for False
```

2. Basic command list
```
    python .\main.py -t  ssd|ssd_andy|haarcascade   # for face detection only
```
