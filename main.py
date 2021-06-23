from skmultiflow.data import ConceptDriftStream, SEAGenerator
from skmultiflow.drift_detection.adwin import ADWIN
from sklearn.linear_model import Perceptron
from fires import FIRES
from online import ONLINE
import warnings

warnings.filterwarnings("ignore")

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
