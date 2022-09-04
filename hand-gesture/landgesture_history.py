import pathlib
import numpy as np
import csv
import copy
import itertools

from collections import deque

class LandMark_History():
    def __init__(self,maxlength = 10):

        self._maxlen = maxlength
        self._landmark_list = deque(maxlen = maxlength)
        for i in range(maxlength):
            self._landmark_list.append(None)


    def update(self,image,results):
        if results.multi_hand_landmarks is not None:

            self._numhands = len(results.multi_hand_landmarks)

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):

                landmark = LandMarks(image,hand_landmarks)
                self._landmark_list.append(landmark)


    def return_array(self):
        export_landmark = []
        if None not in self._landmark_list:
            for i in range(self._maxlen):
                arr = self._landmark_list[i].export()
                export_landmark.append(arr)

            array = np.stack(export_landmark).flatten()

        return array

    def export(self,file,number):
        export_files = []
        if None not in self._landmark_list:
            for i in range(self._maxlen):
                arr = self._landmark_list[i].export()
                # arr = np.insert(arr,0,str(i))
                export_files.append(arr)

            array = np.stack(export_files, axis=0).flatten()
            # print(len(array))
            array_with_index = np.insert(array,0,number)
            # print(len(array))
            # if None not in array:
            with open(file,'a',newline="") as f:
                writer = csv.writer(f)
                writer.writerow([*array_with_index])

    def export_landmark(self):
        object_list = []
        for i in range(self._maxlen):
            object_list.append(self._landmark_list[i]._landmark)

        return object_list

    def export_single_landmark(self,i):
        return self._landmark_list[i]
class LandMarks:
    """
    Todo: Need to get this working for multiland, currently I think it onlys works for one hands
    """
    def __init__(self,image,landmarks):
        self._landmark = landmarks
        self._image = image
        self._image_height = self._image.shape[0]
        self._image_width = self._image.shape[1]
        self._landmark_point = []

        self.normalizes_coordinates()
        self._landmark_rel_points = self.pre_process_landmark(self._landmark_point)
    def normalizes_coordinates(self):
        for _, landmark in enumerate(self._landmark.landmark):
#             print(landmark.x)
            landmark_x = min(int(landmark.x * self._image_width), self._image_width - 1)
            landmark_y = min(int(landmark.y * self._image_height), self._image_height - 1)
            self._landmark_point.append([landmark_x, landmark_y])

    @staticmethod
    def pre_process_landmark(land_mark):
        temp_landmark_list = copy.deepcopy(land_mark)

        # Convert to relative coordinates
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return np.array(temp_landmark_list).reshape((-1,2))

    def export(self):
        result_arr = np.array(self._landmark_rel_points)

        return result_arr.flatten()

    def centeroid(self):
        labels_2_avg = [0,2,5,17]
        x = []
        y = []
        for i,points in enumerate(self._landmark_point):
            if i in labels_2_avg:
                x.append(points[0])
                y.append(points[1])

        self.center = (int(sum(x)/len(x)),int(sum(y)/len(y)))
        return self.center
