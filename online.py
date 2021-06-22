import numpy as np
import shap


class ONLINE:
    def __init__(self, stream, drift_detector, predictor, fires_model=None, do_normalize=False):
        self.window_size = 300
        self.batch_size = 256
        self.stream = stream
        self.drift_detector = drift_detector
        self.predictor = predictor
        self.fires_model = fires_model
        self.do_normalize = do_normalize

        x, y = self.stream.next_sample(self.batch_size)
        self.current_min = np.min(x, axis=0)
        self.current_max = np.max(x, axis=0)
        if self.do_normalize:
            x = self.normalize(x)
        self.window_x = x
        self.window_y = y

        self.predictor.fit(self.window_x, self.window_y)

    def run(self):
        ctr_outer = 0
        while self.stream.has_more_samples():
            y_pred = self.predictor.predict(self.window_x)

            if self.fires_model:
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
                if self.do_normalize:
                    x = self.normalize(x)
            except ValueError:
                break
            self.window_x = np.concatenate((self.window_x, x), axis=0)[-self.window_size:]
            self.window_y = np.concatenate((self.window_y, y))[-self.window_size:]
            ctr_outer += 1

        self.stream.restart()

    def normalize(self, x):
        return (x - self.current_min) / (self.current_max - self.current_min)
