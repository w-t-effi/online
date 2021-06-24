import numpy as np
import shap
import matplotlib.pyplot as plt
from skmultiflow.data.base_stream import Stream
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from fires import FIRES
from utils import get_kdd_conceptdrift_feature_names


class ONLINE:
    """
    Online Normalization and Linear Explanations.
    """
    def __init__(self, stream, drift_detector, predictor, fires_model=None, do_normalize=False, remove_outliers=False):
        """
        Initializes the ONLinE evaluation.

        Args:
            stream (Stream): the data stream
            drift_detector (BaseDriftDetector): the drift detector
            predictor: the predictor
            fires_model (FIRES): the FIRES model (None if SHAP should be used)
            do_normalize (bool): True if the data should be normalised, False otherwise
            remove_outliers (bool): True if outliers should be removed, False otherwise
        """
        self.stream = stream
        self.drift_detector = drift_detector
        self.predictor = predictor
        self.fires_model = fires_model
        self.do_normalize = do_normalize
        self.remove_outliers = remove_outliers
        self.window_size = 300
        self.batch_size = 256
        self.n_frames_explanations = 30

        self.n_frames_initial = 10
        self.k = 150
        x, y = self.stream.next_sample(self.batch_size)

        self.current_min = np.min(x, axis=0)
        self.current_max = np.max(x, axis=0)
        self.current_mean = np.mean(x, axis=0)
        self.current_std = np.std(x, axis=0)

        if self.do_normalize:
            x = self.normalize(x)
        self.window_x = x
        self.window_y = y
        if self.remove_outliers:
            self.window_x, self.window_y = self.remove_outlier_class_sensitive(self.window_x, self.window_y)

        if self.fires_model:
            self.window_x = self.run_fires()

        self.predictor.fit(self.window_x, self.window_y)

    def run(self):
        """
        Executes the ONLinE evaluation.
        """
        ctr_outer = 0
        last_drift_outer = 0
        last_drift_inner = 0
        while self.stream.has_more_samples():
            y_pred = self.predictor.predict(self.window_x)

            if not self.fires_model:
                if ctr_outer % self.n_frames_explanations == 0 and ctr_outer != 0:
                    self.run_shap(ctr_outer)

            ctr_inner = 0
            for i in range(len(self.window_y)):
                self.drift_detector.add_element(self.window_y[i] == y_pred[i])

                if self.drift_detector.detected_change():
                    print(f'Change detected at index: {ctr_outer}.{ctr_inner}')

                    if ctr_outer - last_drift_outer == 1:
                        self.k = self.window_size - last_drift_inner + ctr_inner
                    elif ctr_outer == last_drift_outer:
                        self.k = ctr_inner - last_drift_inner
                    else:
                        self.k = self.window_size
                    drift_window = self.window_x[ctr_inner:ctr_inner + self.k, :]
                    self.current_max = np.max(drift_window, axis=0)
                    self.current_min = np.min(drift_window, axis=0)
                    last_drift_inner = ctr_inner
                    last_drift_outer = ctr_outer
                ctr_inner += 1

            try:
                x, y = self.stream.next_sample(self.batch_size)
                if self.do_normalize:
                    x = self.normalize(x)
            except ValueError:
                break
            self.window_x = np.concatenate((self.window_x, x), axis=0)[-self.window_size:]
            self.window_y = np.concatenate((self.window_y, y))[-self.window_size:]
            if self.remove_outliers:
                self.window_x, self.window_y = self.remove_outlier_class_sensitive(self.window_x, self.window_y)

            if self.fires_model:
                self.window_x = self.run_fires()

            self.predictor.partial_fit(self.window_x, self.window_y)
            ctr_outer += 1

        self.stream.restart()

    def run_fires(self):
        """
        Runs the FIRES feature selection.

        Returns:
            np.ndarray: the data slice with the selected features
        """
        ftr_weights = self.fires_model.weigh_features(self.window_x, self.window_y)
        ftr_selection = np.argsort(ftr_weights)[::-1][:10]
        # TODO visualize feature weights
        x_reduced = np.zeros(self.window_x.shape)
        x_reduced[:, ftr_selection] = self.window_x[:, ftr_selection]
        return x_reduced

    def run_shap(self, time_step):
        """
        Runs the shap explainer and shows the evaluation plots.

        Args:
            time_step (int): the current time step of the evaluation
        """
        explainer = shap.Explainer(self.predictor, self.window_x, feature_names=get_kdd_conceptdrift_feature_names())
        shap_values = explainer(self.window_x)
        plt.title(f'Time step: {time_step}, n samples: {self.window_x.shape[0]}')
        shap.plots.beeswarm(shap_values, plot_size=(20, 10), show=True)
        # plt.savefig(f'plots/shap{time_step}.png')

    def normalize(self, x):
        """
        Normalizes the feature vector.

        Args:
            x (np.ndarray): the feature vector

        Returns:
            np.ndarray: the normalized feature vector
        """
        return (x - self.current_min) / (self.current_max - self.current_min + 10e-99)

    # def remove_outlier(self, x, y):
    #     idx = np.linalg.norm(x - self.current_mean, axis=1) < np.linalg.norm(3 * self.current_std)
    #
    #     return x[idx], y[idx]

    def remove_outlier_class_sensitive(self, x, y):
        """
        Removes the outliers in a class sensitive fashion.

        Args:
            x (np.ndarray): the feature vector
            y (np.ndarray): the labels of the feature vector

        Returns:
            (np.ndarray, np.ndarray): the feature vector and the labels without outliers
        """
        x_0 = x[y == 0]
        x_1 = x[y == 1]
        mean_0 = np.mean(x_0, axis=0)
        std_0 = np.std(x_0, axis=0)

        mean_1 = np.mean(x_1, axis=0)
        std_1 = np.std(x_1, axis=0)

        bools_0 = np.linalg.norm(x_0 - mean_0, axis=1) < np.linalg.norm(3 * std_0)
        bools_1 = np.linalg.norm(x_1 - mean_1, axis=1) < np.linalg.norm(3 * std_1)
        bools = np.ones_like(y)
        bools[y == 0] = bools_0
        bools[y == 1] = bools_1

        return x[bools == 1], y[bools == 1]
