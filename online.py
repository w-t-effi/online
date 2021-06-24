import numpy as np
import shap
import matplotlib.pyplot as plt


class ONLINE:
    def __init__(self, stream, drift_detector, predictor, fires_model=None, do_normalize=False, remove_outliers=False):
        self.stream = stream
        self.drift_detector = drift_detector
        self.predictor = predictor
        self.fires_model = fires_model
        self.do_normalize = do_normalize
        self.remove_outliers = remove_outliers
        self.window_size = 300
        self.batch_size = 256
        self.n_frames_initial = 10

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

        self.explanations_window_x = self.window_x
        self.explanations_window_y = self.window_y

        self.predictor.fit(self.window_x, self.window_y)

    def run(self):
        ctr_outer = 0
        while self.stream.has_more_samples():
            y_pred = self.predictor.predict(self.window_x)

            if ctr_outer == self.n_frames_initial:
                self.run_fires(ctr_outer) if self.fires_model else self.run_shap(ctr_outer)
                self.explanations_window_x = self.explanations_window_x[-self.batch_size:]
                self.explanations_window_y = self.explanations_window_y[-self.batch_size:]

            ctr_inner = 0
            for i in range(len(self.window_y)):
                self.drift_detector.add_element(self.window_y[i] == y_pred[i])

                if self.drift_detector.detected_change():
                    print(f'Change detected at index: {ctr_outer}.{ctr_inner}')
                    self.run_fires(ctr_outer) if self.fires_model else self.run_shap(ctr_outer)
                    self.explanations_window_x = self.explanations_window_x[-self.batch_size:]
                    self.explanations_window_y = self.explanations_window_y[-self.batch_size:]
                    self.current_max = np.max(self.window_x, axis=0)
                    self.current_min = np.min(self.window_x, axis=0)
                    self.current_mean = np.mean(self.window_x, axis=0)
                    self.current_std = np.std(self.window_x, axis=0)
                ctr_inner += 1

            try:
                x, y = self.stream.next_sample(self.batch_size)
                if self.do_normalize:
                    x = self.normalize(x)
            except ValueError:
                break
            self.window_x = np.concatenate((self.window_x, x), axis=0)[-self.window_size:]
            self.window_y = np.concatenate((self.window_y, y))[-self.window_size:]
            self.explanations_window_x = np.concatenate((self.explanations_window_x, x), axis=0)
            self.explanations_window_y = np.concatenate((self.explanations_window_y, y))
            if self.remove_outliers:
                self.window_x, self.window_y = self.remove_outlier_class_sensitive(self.window_x, self.window_y)
                self.explanations_window_x, self.explanations_window_y = self.remove_outlier_class_sensitive(self.explanations_window_x, self.explanations_window_y)
            ctr_outer += 1

        self.stream.restart()

    def run_fires(self, time_step):
        ftr_weights = self.fires_model.weigh_features(self.explanations_window_x, self.explanations_window_y)
        ftr_selection = np.argsort(ftr_weights)[::-1][:10]
        x_reduced = np.zeros(self.explanations_window_x.shape)
        x_reduced[:, ftr_selection] = self.explanations_window_x[:, ftr_selection]

    def run_shap(self, time_step):
        explainer = shap.Explainer(self.predictor, self.explanations_window_x, feature_names=self.stream.feature_names)
        shap_values = explainer(self.explanations_window_x)
        shap.plots.beeswarm(shap_values, show=False)
        plt.savefig(f'test{time_step}.png')

    def normalize(self, x):
        return (x - self.current_min) / (self.current_max - self.current_min + 10e-99)

    def remove_outlier(self, x, y):
        idx = np.linalg.norm(x - self.current_mean, axis=1) < np.linalg.norm(3 * self.current_std)

        return x[idx], y[idx]

    def remove_outlier_class_sensitive(self, x, y):
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
