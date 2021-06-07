import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skmultiflow.data import ConceptDriftStream, SEAGenerator
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.linear_model import Perceptron
import warnings
warnings.filterwarnings("ignore")

alternate1 = ConceptDriftStream(
    stream=SEAGenerator(balance_classes=False, classification_function=1, random_state=112, noise_percentage=0.1),
    position=50000,
    width=1,
    random_state=0)

stream = ConceptDriftStream(
    stream=SEAGenerator(balance_classes=False, classification_function=0, random_state=112, noise_percentage=0.1),
    drift_stream=alternate1,
    position=50000,
    width=1,
    random_state=0)

adwin = ADWIN()
classifier = Perceptron()

window_size = 300
batch_size = 256

x, y = stream.next_sample(batch_size)
window_x = x
window_y = y
classifier.fit(window_x, window_y)
ctr = 0
while stream.has_more_samples():
    y_pred = classifier.predict(window_x)

    for i in range(len(y)):
        adwin.add_element(y[i] == y_pred[i])
        if adwin.detected_change():
            print(f'Change detected at index: {ctr}')

    try:
        x, y = stream.next_sample(batch_size)
    except ValueError:
        break
    window_x = np.concatenate((window_x, x), axis=0)[-window_size:]
    window_y = np.concatenate((window_y, y))[-window_size:]
    ctr += 1

stream.restart()
