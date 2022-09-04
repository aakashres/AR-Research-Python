import re


class Buffer:
    buffer_length = 10
    frames = []

    def __init__(self, buffer_length):
        self.buffer_length = buffer_length

    def is_buffer_full(self):
        return len(self.frames) == self.buffer_length

    def is_buffer_empty(self):
        return len(self.frames) == 0

    def add_frames(self, frame):
        if not self.is_buffer_full():
            self.frames.append(frame)
        return False

    def get_frames(self):
        if not self.is_buffer_empty() and self.is_buffer_full():
            return self.frames.pop(0)
        return False

    def change_frame_buffer_size(self, update_type, update_length):
        self.buffer_length = self.buffer_length + update_type * update_length

    def get_buffer_frames(self):
        return self.frames.copy()

    def update_frame(self, frame, i):
        self.frames[i] = frame
