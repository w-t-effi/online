from skmultiflow.data import FileStream
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import shap
from normalize import normalize
from utils import get_spambase_feature_names

shap.initjs()

normalize_data = False

feature_names = get_spambase_feature_names()

stream = FileStream('data/spambase.csv', target_idx=0)
x, y = stream.next_sample(batch_size=100)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

if normalize_data:
    x_train = normalize(x_train)
    x_test = normalize(x_test)

predictor = Perceptron()
predictor.partial_fit(x_train, y_train, stream.target_values)
explainer = shap.Explainer(predictor, x_test, feature_names=feature_names)
shap_values = explainer(x_test)

shap.plots.beeswarm(shap_values)
