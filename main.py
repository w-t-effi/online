import numpy as np
from skmultiflow.data import ConceptDriftStream, SEAGenerator
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.linear_model import Perceptron
from fires import FIRES
import warnings
import shap

warnings.filterwarnings("ignore")


class ONLINE:
    def __init__(self, stream, drift_detector, predictor, fires_model=None):
        self.window_size = 300
        self.batch_size = 256
        self.stream = stream
        self.drift_detector = drift_detector
        self.predictor = predictor
        self.fires_model = fires_model

        x, y = self.stream.next_sample(self.batch_size)
        self.current_min = np.min(x, axis=0)
        self.current_max = np.max(x, axis=0)
        x = self.normalize(x)
        self.window_x = x
        self.window_y = y

        self.predictor.fit(self.window_x, self.window_y)

    def run(self):
        ctr_outer = 0
        while self.stream.has_more_samples():
            y_pred = self.predictor.predict(self.window_x)

            if fires_model:
                ftr_weights = self.fires_model.weigh_features(self.window_x, self.window_y)
                ftr_selection = np.argsort(ftr_weights)[::-1][:10]
                x_reduced = np.zeros(self.window_x.shape)
                x_reduced[:, ftr_selection] = self.window_x[:, ftr_selection]
                self.window_x = x_reduced
            else:
                explainer = shap.Explainer(self.predictor, self.window_x)
                shap_values = explainer(self.window_x)
                shap.plots.beeswarm(shap_values)

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

fires_model = FIRES(n_total_ftr=concept_drift_stream.n_features,  # Total no. of features
                    target_values=concept_drift_stream.target_values,  # Unique target values (class labels)
                    mu_init=0,  # Initial importance parameter
                    sigma_init=1,  # Initial uncertainty parameter
                    penalty_s=0.01,  # Penalty factor for the uncertainty (corresponds to gamma_s in the paper)
                    penalty_r=0.01,  # Penalty factor for the regularization (corresponds to gamma_r in the paper)
                    epochs=1,  # No. of epochs that we use each batch of observations to update the parameters
                    lr_mu=0.01,  # Learning rate for the gradient update of the importance
                    lr_sigma=0.01,  # Learning rate for the gradient update of the uncertainty
                    scale_weights=True,  # If True, scale feature weights into the range [0,1]
                    model='probit')  # Name of the base model to compute the likelihood

adwin = ADWIN()
perceptron = Perceptron()

online = ONLINE(concept_drift_stream, adwin, perceptron, fires_model)
online.run()
