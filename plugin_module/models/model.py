import os


import matplotlib.pyplot as plt
import librosa.display
import numpy as np

from keras.utils import load_img, img_to_array
import keras


# Model dir
# MODEL_DIR = "./models/audio_classification.hdf5"
MODEL_DIR = "./models/model_LSTM.h5"


class LSTM_AUDIO:
    def __init__(self):
        self.model = keras.models.load_model(MODEL_DIR)

    def load_audio_file(self):

        # Remove cache
        if (os.path.exists("../cache/output.wav")):
            os.remove("../cache/output.wav")

        # LSTM
        features = self.features_extractor("./cache/output.wav")

        preds = self.predict_class(features)[0]

        # Remove cache
        if (os.path.exists("../cache/output.wav")):
            os.remove("../cache/output.wav")

        if (preds <= 0.3):
            return "No Abuse Detected"

        elif (preds > 0.3 and preds <= 0.65):
            return "Warning!!!"

        else:
            return "Abuse Detected!!!"

        # # CNN
        # self.create_spectrogram("./cache/angry_011.wav", "./cache/output.png")
        # images = []
        # images.append(img_to_array(
        #     load_img("./cache/output.png", target_size=(224, 224, 3))))
        # images = np.array(images)
        # model = keras.models.load_model("./models/model.h5")
        # print(model.predict(images))

    def create_spectrogram(self, audio_file, image_file):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        y, sr = librosa.load(audio_file)
        ms = librosa.feature.melspectrogram(y=y, sr=sr)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        librosa.display.specshow(log_ms, sr=sr)

        fig.savefig(image_file)
        plt.close(fig)

    def predict_class(self, features):
        features_reshaped = np.reshape(features, (1, 1, 40))
        return self.model.predict(features_reshaped)

    def features_extractor(self, file):
        # load the file (audio)
        # audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        audio, sample_rate = librosa.load(file)

        # we extract mfcc
        mfccs_features = librosa.feature.mfcc(
            y=audio, sr=sample_rate, n_mfcc=40)
        # in order to find out scaled feature we do mean of transpose of value
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features
