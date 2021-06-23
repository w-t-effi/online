import numpy as np
from skmultiflow.data import ConceptDriftStream, SEAGenerator
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.linear_model import Perceptron
import warnings
from fires import FIRES
from skmultiflow.data.file_stream import FileStream

warnings.filterwarnings("ignore")


def normalize(x, current_min, current_max):
    return (x - current_min) / (current_max - current_min)


stream = FileStream('data/synthetic/sea')
adwin = ADWIN()
classifier = Perceptron()

window_size = 300
batch_size = 256

x, y = stream.next_sample(batch_size)
window_x = x
window_y = y
current_min = np.min(window_x, axis=0)
current_max = np.max(window_x, axis=0)
current_mean = np.mean(window_x, axis=0)
x = normalize(x, current_min, current_max)

classifier.fit(window_x, window_y)
ctr = 0

n_selected_ftr = 3

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

while stream.has_more_samples():
    y_pred = classifier.predict(window_x)

    ctr_2 = 0
    for i in range(len(y)):
        adwin.add_element(y[i] == y_pred[i])

        if adwin.detected_change():
            print(f'Change detected at index: {ctr}.{ctr_2}')
            current_max = np.max(window_x, axis=0)
            current_min = np.min(window_x, axis=0)
            current_mean = np.mean(window_x, axis=0)
        ctr_2 += 1

    try:
        x, y = stream.next_sample(batch_size)
        #x = normalize(x, current_min, current_max)
    except ValueError:
        break

    ###FIRES###
    # Select features
    ftr_weights = fires_model.weigh_features(x, y)  # Get feature weights with FIRES
    ftr_selection = np.argsort(ftr_weights)[::-1][:n_selected_ftr]

    # Truncate x (retain only selected features, 'remove' all others, e.g. by replacing them with 0)
    x_reduced = np.zeros(x.shape)
    x_reduced[:, ftr_selection] = x[:, ftr_selection]
    x=x_reduced
    ###########

    window_x = np.concatenate((window_x, x), axis=0)[-window_size:]
    window_y = np.concatenate((window_y, y))[-window_size:]
    ctr += 1

stream.restart()

# def compute_min_max(delta, window_x):
#     _min = np.min(window_x, axis = 0)
#     _max = np.max(window_x, axis = 0)
#
#     if delta<np.abs(np.mean(window_x, axis=0)-current_mean)/current_mean:
#         current_min = _min
#         current_max = _max
