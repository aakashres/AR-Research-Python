# Hand Gesture Modules


The following describes the hand gesture modules constructed using tensorflow lite. The gesture recording models currently have 4 recorded hand gesture, with more can be added using the included training notebook ("keypoint_history_classification.ipynb")


# To train the new models
There are 4 modes:
1. Classifications
2. Pixelation
3. Creating Training data
4. Classifications and Pixelation together

`python main.py --mode <modenumber>`
## To perform Classifications:

`python main.py --mode 1`

## For mode 3, there are a number optional parameters needed to specify the gesture id, this is done as follow:

`python main.py --mode 3 --number <gesture_id>`




# Sources
[1] 高橋かずひと(https://twitter.com/KzhtTkhs)
