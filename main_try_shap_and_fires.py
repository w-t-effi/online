import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skmultiflow.data import ConceptDriftStream, SEAGenerator
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.linear_model import Perceptron
import warnings
import shap
from fires import FIRES
warnings.filterwarnings("ignore")

def normalize(x, current_min, current_max):
    return (x - current_min) / (current_max - current_min)

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

fires_model = FIRES(n_total_ftr=stream.n_features,          # Total no. of features
                    target_values=stream.target_values,     # Unique target values (class labels)
                    mu_init=0,                              # Initial importance parameter
                    sigma_init=1,                           # Initial uncertainty parameter
                    penalty_s=0.01,                         # Penalty factor for the uncertainty (corresponds to gamma_s in the paper)
                    penalty_r=0.01,                         # Penalty factor for the regularization (corresponds to gamma_r in the paper)
                    epochs=1,                               # No. of epochs that we use each batch of observations to update the parameters
                    lr_mu=0.01,                             # Learning rate for the gradient update of the importance
                    lr_sigma=0.01,                          # Learning rate for the gradient update of the uncertainty
                    scale_weights=True,                     # If True, scale feature weights into the range [0,1]
                    model='probit')                         # Name of the base model to compute the likelihood


window_size = 300
batch_size = 256

x, y = stream.next_sample(batch_size)
window_x = x
window_y = y
classifier.fit(window_x, window_y)
ctr = 0

use_normalization = True
use_feat_det = True
shap_rate = 25

current_min = np.min(window_x,axis=0)
current_max = np.max(window_x,axis=0)
current_mean = np.mean(window_x,axis=0)
while stream.has_more_samples():
    y_pred = classifier.predict(window_x)

    for i in range(len(y)):
        adwin.add_element(y[i] == y_pred[i])
        if adwin.detected_change():
            print(f'Change detected at index: {ctr}')
            current_max=np.max(window_x,axis=0)
            current_min=np.min(window_x,axis=0)
            current_mean=np.mean(window_x,axis=0)

    try:
        x, y = stream.next_sample(batch_size)
        x = normalize(x, current_min, current_max)
    except ValueError:
        break

    explainer = shap.explainers.Exact(classifier._predict_proba_lr, window_x)
    shap_values = explainer(window_x[:100])
    window_x = np.concatenate((window_x, x), axis=0)[-window_size:]
    window_y = np.concatenate((window_y, y))[-window_size:]
    ctr += 1
    shap.plots.bar(explainer(window_x)[...,1])
stream.restart()


def compute_min_max(delta):
    _min = np.min(window_x, axis = 0)
    _max = np.max(window_x, axis = 0)
    
    if(delta<=np.abs(np.mean(window_x, axis=0)-current_mean)/current_mean):
        current_min = _min
        current_max = _max


