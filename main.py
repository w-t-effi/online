import numpy as np
from skmultiflow.data import ConceptDriftStream, SEAGenerator
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.linear_model import Perceptron
import warnings

warnings.filterwarnings("ignore")


class ONLINE:
    def __init__(self, stream, drift_detector, predictor):
        self.window_size = 300
        self.batch_size = 256
        self.stream = stream
        self.drift_detector = drift_detector
        self.predictor = predictor

        x, y = self.stream.next_sample(self.batch_size)
        self.current_min = np.min(x, axis=0)
        self.current_max = np.max(x, axis=0)
        x = self.normalize(x)
        self.window_x = x
        self.window_y = y

        predictor.fit(self.window_x, self.window_y)

    def run(self):
        ctr_outer = 0
        while self.stream.has_more_samples():
            y_pred = self.predictor.predict(self.window_x)

            ctr_inner = 0
            for i in range(len(self.window_y)):
                self.drift_detector.add_element(self.window_y[i] == y_pred[i])

                if self.drift_detector.detected_change():
                    print(f'Change detected at index: {ctr_outer}.{ctr_inner}')
                    self.current_max = np.max(self.window_x, axis=0)
                    self.current_min = np.min(self.window_x, axis=0)
                ctr_inner += 1

            try:
                x, y = self.stream.next_sample(self.batch_size)
                x = self.normalize(x)
            except ValueError:
                break
            self.window_x = np.concatenate((self.window_x, x), axis=0)[-self.window_size:]
            self.window_y = np.concatenate((self.window_y, y))[-self.window_size:]
            ctr_outer += 1

        self.stream.restart()

    def normalize(self, x):
        return (x - self.current_min) / (self.current_max - self.current_min)


alternate1 = ConceptDriftStream(
    stream=SEAGenerator(balance_classes=False, classification_function=1, random_state=112, noise_percentage=0.1),
    position=50000,
    width=1,
    random_state=0)

concept_drift_stream = ConceptDriftStream(
    stream=SEAGenerator(balance_classes=False, classification_function=0, random_state=112, noise_percentage=0.1),
    drift_stream=alternate1,
    position=50000,
    width=1,
    random_state=0)

adwin = ADWIN()
perceptron = Perceptron()

online = ONLINE(concept_drift_stream, adwin, perceptron)
online.run()
