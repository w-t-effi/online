import os
from datetime import datetime
import numpy as np
import shap
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skmultiflow.data.file_stream import FileStream
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from sklearn.metrics import accuracy_score
from fires import FIRES
from utils import get_kdd_conceptdrift_feature_names


class ONLINE:
    """
    Online Normalization and Linear Explanations.
    """

    def __init__(self, stream, drift_detector, predictor, fires_model=None, do_normalize=False, remove_outliers=False,
                 delta=0.25, y_drift_detection=False, gamma=0):
        """
        Initializes the ONLinE evaluation.

        Args:
            stream (FileStream): the data stream
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
        self.y_drift_detection = y_drift_detection
        self.remove_outliers = remove_outliers
        self.window_size = 300
        self.batch_size = 256
        self.n_frames_explanations = 30
        self.gamma = gamma
        self.n_selected_features = 10 if self.stream.filename == 'kdd_conceptdrift' or self.stream.filename == 'spambase' else 4
        self.shap_top_features = []
        self.predictor_top_features = []
        self.accuracy_scores = []

        self.feature_names = np.array(
            get_kdd_conceptdrift_feature_names()) if self.stream.filename == 'kdd_conceptdrift' else np.array(
            self.stream.feature_names)

        self.delta = delta
        x, y = self.stream.next_sample(self.batch_size)

        x_filtered = self.remove_outlier_class_sensitive(x, y) if self.remove_outliers else x

        self.current_mean = np.mean(x_filtered, axis=0)
        self.current_std = np.std(x_filtered, axis=0)

        if self.do_normalize:
            x = self.normalize(x)
        self.window_x = x
        self.window_y = y

        if self.fires_model:
            self.window_x = self.run_fires(0)

        self.predictor.fit(self.window_x, self.window_y)
        time_obj = datetime.now()
        time_str = f'{time_obj.year}-{time_obj.month}-{time_obj.day}_{time_obj.hour}-{time_obj.minute}-{time_obj.second}'

        self.dir_path = f'plots/{time_str}/'
        os.makedirs(self.dir_path)
        with open(self.dir_path + 'params.txt', "w") as file:
            file.write(
                f'FIRES: {self.fires_model}\nNormalize: {self.do_normalize}\nRemove outliers: {self.remove_outliers}\nData: {stream.filename}')

    def run(self):
        """
        Executes the ONLinE evaluation.
        """
        ctr_outer = 0
        self.last_drift_outer = 0
        self.last_drift_inner = 0
        while self.stream.has_more_samples():
            y_pred = self.predictor.predict(self.window_x)
            self.accuracy_scores.append(accuracy_score(self.window_y, y_pred))

            if not self.fires_model:
                self.run_shap(ctr_outer)

            ctr_inner = 0
            try:
                x, y = self.stream.next_sample(self.batch_size)
                if ctr_outer == 50 or ctr_outer == 51:
                    x = self.force_concept_drift(x, k=1000, i=2)

            except ValueError:
                break
            if self.y_drift_detection:
                for i in range(len(self.window_y)):
                    self.drift_detector.add_element(self.window_y[i] == y_pred[i])

                    if self.drift_detector.detected_change():
                        print(f'Change detected at index: {ctr_outer}.{ctr_inner}')
                        self.update_statistics(x, y)
                    ctr_inner += 1

            else:
                self.detect_concept_drift_x(x, y, ctr_outer, ctr_inner)
                ctr_inner += 1

            if self.do_normalize:
                x = self.normalize(x)

            self.window_x = np.concatenate((self.window_x, x), axis=0)[-self.window_size:]
            self.window_y = np.concatenate((self.window_y, y))[-self.window_size:]

            if self.fires_model:
                self.window_x = self.run_fires(ctr_outer)

            self.predictor.partial_fit(self.window_x, self.window_y)
            ftr_weights = self.predictor.coef_[0]
            ftr_selection = np.argsort(ftr_weights)[::-1][:self.n_selected_features]
            self.predictor_top_features.append(ftr_selection)
            if ctr_outer % self.n_frames_explanations == 0 and ctr_outer != 0:
                title = f'Prediction weights. Time step: {ctr_outer}, n samples: {self.window_x.shape[0]}'
                path_name = f'prediction{ctr_outer}.png'
                self.draw_selection_plot(ftr_weights, title, path_name, self.feature_names)

            ctr_outer += 1

        selection = self.fires_model.selection if self.fires_model else self.shap_top_features
        title = 'FIRES top features' if self.fires_model else 'SHAP top features'
        path_name = 'fires_top.png' if self.fires_model else 'shap_top.png'

        self.draw_top_features_plot(selection, title, path_name, self.feature_names)
        self.draw_top_features_plot(self.predictor_top_features, 'Predictor top features', 'predictor_top.png',
                                    self.feature_names)
        self.draw_accuracy()
        self.stream.restart()

    def run_fires(self, time_step):
        """
        Runs the FIRES feature selection.

        Returns:
            np.ndarray: the data slice with the selected features
        """
        ftr_weights = self.fires_model.weigh_features(self.window_x, self.window_y)
        ftr_selection = np.argsort(ftr_weights)[::-1][:self.n_selected_features]
        self.fires_model.selection.append(ftr_selection.tolist())
        if time_step % self.n_frames_explanations == 0 and time_step != 0:
            title = f'FIRES weights. Time step: {time_step}, n samples: {self.window_x.shape[0]}'
            path_name = f'fires{time_step}.png'
            self.draw_selection_plot(ftr_weights, title, path_name, feature_names=self.feature_names)
        x_reduced = np.zeros(self.window_x.shape)
        x_reduced[:, ftr_selection] = self.window_x[:, ftr_selection]
        return x_reduced

    def run_shap(self, time_step):
        """
        Runs the shap explainer and shows the evaluation plots.

        Args:
            time_step (int): the current time step of the evaluation
        """
        explainer = shap.Explainer(self.predictor, self.window_x, feature_names=self.feature_names)
        shap_values = explainer(self.window_x)
        ftr_weights = np.abs(shap_values.values).mean(axis=0)
        ftr_selection = np.argsort(ftr_weights)[::-1][:self.n_selected_features]
        self.shap_top_features.append(ftr_selection)
        if time_step % self.n_frames_explanations == 0 and time_step != 0:
            plt.title(f'Shap. Time step: {time_step}, n samples: {self.window_x.shape[0]}')
            shap.plots.beeswarm(shap_values, plot_size=(30, 15), show=False)
            plt.savefig(self.dir_path + f'shap{time_step}.png', bbox_inches='tight')
            plt.close()

    def update_statistics(self, x, y):
        drift_window_x_filtered = self.remove_outlier_class_sensitive(x, y) if self.remove_outliers else x

        former_mean = self.current_mean
        former_std = self.current_std

        if drift_window_x_filtered.shape[0] > 0:
            self.current_mean = np.mean(drift_window_x_filtered, axis=0)
            self.current_std = np.std(drift_window_x_filtered, axis=0)

            # interpolate bw current and previous max/min
            self.current_mean = (1 - self.gamma) * self.current_mean + self.gamma * former_mean
            self.current_std = (1 - self.gamma) * self.current_std + self.gamma * former_std

    def normalize(self, x):
        """
        Normalizes the feature vector.

        Args:
            x (np.ndarray): the feature vector

        Returns:
            np.ndarray: the normalized feature vector
        """
        return (x - self.current_mean) / (self.current_std + 10e-99)

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

        return x[bools == 1]

    def detect_concept_drift_x(self, x, y, ctr_outer, ctr_inner):
        """
        Detects concept drift in feature values and updates statistics if necessary.

        Args:
            x (np.ndarray): the feature vector
            y (np.ndarray): the target vector
            ctr_outer (int): frame count
            ctr_inner (int): sample in batch count
        """

        filtered_data = self.remove_outlier_class_sensitive(x, y) if self.remove_outliers else x
        current_mean = np.mean(filtered_data, axis=0)

        if (np.linalg.norm(np.abs(current_mean - self.current_mean) / (self.current_mean + 1e-10))) > self.delta:
            self.update_statistics(x, y)
            print(f'Change detected at index: {ctr_outer}.{ctr_inner}')

    @staticmethod
    def force_concept_drift(x, k=1000, i=0):
        """
        Scales feature i of x by k to enforce concept drift

        Args:
            x (np.ndarray): the feature vector
            k (int): scaling weight
            i (int): feature index

        Returns:
            np.ndarray: the drifted feature vector
        """
        x[:, i] = x[:, i] * k
        return x

    def draw_top_features_plot(self, top_features, title, path_name, feature_names):
        """
        Draws the most selected features over time.
        """
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.grid(True, axis='y')
        y = [feature for features in top_features for feature in features]
        counts = np.bincount(y)
        top_ftr_idx = counts.argsort()[-self.n_selected_features:][::-1]
        ax.bar(np.arange(self.n_selected_features), counts[top_ftr_idx], width=0.3, zorder=100)
        ax.set_xticklabels(feature_names[top_ftr_idx], rotation=15, ha='right')
        ax.set_ylabel('Times Selected', size=20, labelpad=1.5)
        ax.set_xlabel(f'Top {self.n_selected_features} Features', size=20, labelpad=1.6)
        ax.tick_params(axis='both', labelsize=20 * 0.7, length=0)
        ax.set_xticks(np.arange(self.n_selected_features))
        plt.title(title, size=20)
        plt.savefig(self.dir_path + path_name, bbox_inches='tight')
        plt.close()

    def draw_selection_plot(self, ftr_weights, title, path_name, feature_names):
        """
        Draws the selected features.
        """
        ftr_selection = np.argsort(ftr_weights)[::-1][:self.n_selected_features]
        ftr_selection_vals = ftr_weights[ftr_selection]

        plt.ioff()

        fig, ax = plt.subplots(figsize=(20, 12))
        ax.grid(True, axis='y')
        ax.bar(np.arange(len(ftr_selection)), ftr_selection_vals)
        ax.set_xticks(np.arange(len(ftr_selection)))
        ax.set_xticklabels(feature_names[ftr_selection], rotation=15)
        ax.set_xlabel('Top Features', size=20, labelpad=1.6)
        ax.set_ylabel('Feature Weights', size=20, labelpad=1.5)
        plt.title(title, size=20)
        plt.savefig(self.dir_path + path_name, bbox_inches='tight')
        plt.close()

    def draw_accuracy(self):
        plt.ioff()

        fig, ax = plt.subplots(figsize=(20, 12))
        ax.grid(True, axis='y')
        ax.plot(np.arange(len(self.accuracy_scores)), self.accuracy_scores)
        ax.set_xlabel('Time Step', size=20, labelpad=1.6)
        ax.set_ylabel('Accuracy', size=20, labelpad=1.5)
        plt.title(f'Prediction Accuracy', size=20)
        plt.savefig(self.dir_path + 'prediction_accuracy.png', bbox_inches='tight')
        plt.close()
        import pandas as pd
        df = pd.DataFrame({'Accuracy_scores': self.accuracy_scores})
        df.to_csv(self.dir_path + 'accuracy.csv', index=False)
