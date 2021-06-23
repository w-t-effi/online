import numpy as np
from skmultiflow.data import ConceptDriftStream, SEAGenerator
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.linear_model import Perceptron
import warnings
import shap
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


shap_rate = 450

x, y = stream.next_sample(batch_size)
window_x = x
window_y = y
current_min = np.min(window_x, axis=0)
current_max = np.max(window_x, axis=0)
current_mean = np.mean(window_x, axis=0)
x = normalize(x, current_min, current_max)

classifier.fit(window_x, window_y)
ctr = 0



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
    window_x = np.concatenate((window_x, x), axis=0)[-window_size:]
    window_y = np.concatenate((window_y, y))[-window_size:]

    

    ###SHAP###
    
    if(ctr%shap_rate==0):
        explainer = shap.explainers.Exact(classifier._predict_proba_lr, window_x)
        shap_values = explainer(window_x)
        shap.plots.bar(explainer(window_x)[...,1])
    ##########
    ctr += 1

stream.restart()

# def compute_min_max(delta, window_x):
#     _min = np.min(window_x, axis = 0)
#     _max = np.max(window_x, axis = 0)
#
#     if delta<np.abs(np.mean(window_x, axis=0)-current_mean)/current_mean:
#         current_min = _min
#         current_max = _max
