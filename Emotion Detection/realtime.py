# import the opencv library
import cv2


import tensorflow as tf
import numpy as np
from PIL import Image


class TensorflowLiteClassificationModel:
    def __init__(self, model_path, labels, image_size=48):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        self.labels = [line.strip() for line in open(labels, "r")]
        self.image_size = image_size

    def run_from_filepath(self, image_path):
        input_data_type = self._input_details[0]["dtype"]
        image = np.array(
            Image.open(image_path).resize((self.image_size, self.image_size)),
            dtype=input_data_type,
        )
        if input_data_type == np.float32:
            image = image / 255.0

        if image.shape == (1, 48, 48):
            image = np.stack(image * 3, axis=0)

        return self.run(image)

    def run(self, image):
        """
        args:
          image: a (1, image_size, image_size, 3) np.array

        Returns list of [Label, Probability], of type List<str, float>
        """

        self.interpreter.set_tensor(self._input_details[0]["index"], image)
        self.interpreter.invoke()
        tflite_interpreter_output = self.interpreter.get_tensor(
            self._output_details[0]["index"]
        )
        probabilities = np.array(tflite_interpreter_output[0])
        print(probabilities)
        # create list of ["label", probability], ordered descending probability
        label_to_probabilities = []
        for i, probability in enumerate(probabilities):
            label_to_probabilities.append([self.labels[i], float(probability)])
        return sorted(label_to_probabilities, key=lambda element: element[1])


face_detector = cv2.CascadeClassifier(
    "./face_models/haarcascade_frontalface_default.xml"
)
emotion_model = TensorflowLiteClassificationModel(
    "./model/model300.tflite", "./model/labelmap.txt"
)
# define a video capture object
vid = cv2.VideoCapture()
vid.open(0, cv2.CAP_DSHOW)
if not vid.isOpened():
    print("Could not open video")
    exit()

while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    if not ret:
        print("end of the video file")
        break

    # convert image to gray scale of each frames
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces in the frame
    faces = face_detector.detectMultiScale(gray_frame, 1.3, 5)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cropped_face = frame[y : y + h, x : x + w]
        test_image = cv2.resize(cropped_face, (48, 48))
        test_image = test_image.astype("float32")
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)
        print(emotion_model.run(test_image))

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
