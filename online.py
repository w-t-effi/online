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

        self.shap_top_features = []

        self.n_frames_initial = 10
        self.k = 150
        x, y = self.stream.next_sample(self.batch_size)

        x_filtered = self.remove_outlier_class_sensitive(x, y) if self.remove_outliers else x
        self.current_min = np.min(x_filtered, axis=0)
        self.current_max = np.max(x_filtered, axis=0)
        self.current_mean = np.mean(x_filtered, axis=0)
        self.current_std = np.std(x_filtered, axis=0)

        if self.do_normalize:
            x = self.normalize(x)
        self.window_x = x
        self.window_y = y

        if self.fires_model:
            self.window_x = self.run_fires(0)

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

                    drift_window_x = self.window_x[ctr_inner:ctr_inner + self.k, :]
                    drift_window_y = self.window_y[ctr_inner:ctr_inner + self.k]
                    drift_window_x_filtered = self.remove_outlier_class_sensitive(drift_window_x, drift_window_y) if self.remove_outliers else drift_window_x

                    self.current_max = np.max(drift_window_x_filtered, axis=0) if drift_window_x_filtered.shape[0] > 0 else self.current_max
                    self.current_min = np.min(drift_window_x_filtered, axis=0) if drift_window_x_filtered.shape[0] > 0 else self.current_max
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

            if self.fires_model:
                self.window_x = self.run_fires(ctr_outer)

            self.predictor.partial_fit(self.window_x, self.window_y)
            ctr_outer += 1

        self.draw_top_features_plot()
        self.stream.restart()

    def run_fires(self, time_step):
        """
        Runs the FIRES feature selection.

        Returns:
            np.ndarray: the data slice with the selected features
        """
        ftr_weights = self.fires_model.weigh_features(self.window_x, self.window_y)
        ftr_selection = np.argsort(ftr_weights)[::-1][:10]
        self.fires_model.selection.append(ftr_selection.tolist())
        if time_step % self.n_frames_explanations == 0 and time_step != 0:
            self.draw_selection_plot(ftr_weights, time_step)
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
        ftr_weights = explainer.coef
        ftr_selection = np.argsort(ftr_weights)[::-1][:10]
        self.shap_top_features.append(ftr_selection)
        if time_step % self.n_frames_explanations == 0 and time_step != 0:
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

    @staticmethod
    def remove_outlier_class_sensitive(x, y):
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

        return x[bools == 1]  # , y[bools == 1]

    def draw_top_features_plot(self):
        """
        Draws the most selected features over time.

        Args:
            feature_names (list): the list of feature names
        """
        selection = self.fires_model.selection if self.fires_model else self.shap_top_features
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.grid(True, axis='y')
        y = [feature for features in selection for feature in features]
        counts = np.bincount(y)
        top_ftr_idx = counts.argsort()[-10:][::-1]
        ax.bar(np.arange(10), counts[top_ftr_idx], width=0.3, zorder=100)
        ax.set_xticklabels(np.asarray(get_kdd_conceptdrift_feature_names())[top_ftr_idx], rotation=15, ha='right')
        ax.set_ylabel('Times Selected', size=20, labelpad=1.5)
        ax.set_xlabel('Top 10 Features', size=20, labelpad=1.6)
        ax.tick_params(axis='both', labelsize=20 * 0.7, length=0)
        ax.set_xticks(np.arange(10))
        ax.set_xlim(-0.2, 9.2)
        title = "FIRES top features" if self.fires_model else "SHAP top features"
        plt.title(title, size=20)
        plt.show()

    def draw_selection_plot(self, ftr_weights, time_step):
        """
        Draws the selected features.
        """
        ftr_selection = np.argsort(ftr_weights)[::-1][:10]
        ftr_selection_vals = ftr_weights[ftr_selection]
        feature_names = np.array(get_kdd_conceptdrift_feature_names())

        fig, ax = plt.subplots(figsize=(20, 12))
        ax.grid(True, axis='y')

        ax.bar(np.arange(len(ftr_selection)), ftr_selection_vals)
        ax.set_xticks(np.arange(len(ftr_selection)))
        ax.set_xticklabels(feature_names[ftr_selection], rotation=15)
        ax.set_xlabel('Top Features', size=20, labelpad=1.6)
        ax.set_ylabel('Feature Weights', size=20, labelpad=1.5)
        plt.title(f'Time step: {time_step}, n samples: {self.window_x.shape[0]}', size=20)
        plt.show()
