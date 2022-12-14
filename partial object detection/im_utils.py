import os
import cv2

# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

VIDEO_TYPE = {
    "avi": cv2.VideoWriter_fourcc(*"XVID"),
    "mp4": cv2.VideoWriter_fourcc(*"mp4v"),
}


def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


def get_dims(cap, res="1080p"):
    width, height = STD_DIMENSIONS["480p"]
    # if res in STD_DIMENSIONS:
    #     width, height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE["mp4"]
