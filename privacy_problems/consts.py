from pathlib import Path

DETECTORS = ["ssd", "haarcascade", "ssd_andy"]
SSD_IMAGE_TARGET_SIZE = (300, 300)
SSD_MODEL = Path("../models/Res_10/Res_10_300x300.caffemodel")
SSD_PROTOTEXT = Path("../models/Res_10/Res_10_300x300_ssd_iter_140000.txt")
HAARCASCADE_FACE_MODEL = Path("../models/Haarcascade/haarcascade_frontalface_alt.xml")


EMOTION_MODEL = Path("../models/Emotion/emotion_recognition.h5")
EMOTION_TFLITE_MODEL = Path("../models/Emotion/emotion_model.tflite")


TRAINING_PHOTO_COUNT = 5
TRAINING_DATA_PATH = Path("./training_data")
FACE_ENCODING_DATA = Path("./encoding_data/encoding.npy")
FACE_ENCODING_LABEL= Path("./encoding_data/labels.npy")


RECOGNIZER_MODEL_FILE = Path("./recognizer_model/known_faces.xml")
RECOGNIZER_LABEL_FILE = Path("./recognizer_model/labels.db")


DLIB_SHAPE_PREDICTOR_MODEL = Path("../models/DlibFaceRec/shape_predictor_68_face_landmarks.dat")
DLIB_FACE_ENCODE_MODEL = Path("../models/DlibFaceRec/dlib_face_recognition_resnet_model_v1.dat")