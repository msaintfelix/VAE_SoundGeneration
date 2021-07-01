import numpy as np
import librosa

class Loader:
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        # librosa.load returns a tuple, let's ignore the sr by calling [0]
        signal = librosa.load(file_path, sr=self.sample_rate, duration=self.duration, mono=self.mono)[0]
        return signal


class Padder:
    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (num_missing_items, 0), mode=self.mode)
        return padded_array


    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (0, num_missing_items), mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        # for convenience, let's exclude the last sample of stft as its shape is (1 + frame_size/2, num_frames)
        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)[:-1]



class MinMaxNormalizer:
    def __init_(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalize(self, original_min, original_max):


