import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from skmultiflow.data import ConceptDriftStream, \
    AGRAWALGenerator, \
    HyperplaneGenerator, \
    SEAGenerator

def save_to_file(x, y, name):  # save the data being provided
    data = pd.DataFrame(x)
    data['class'] = y.astype(int)
    
    data.to_csv('data/synthetic/'+name+'.csv', index=False, header=False)

# Drift @ 200.000
alternate2 = ConceptDriftStream(
    stream=SEAGenerator(balance_classes=False, classification_function=2, random_state=112, noise_percentage=0.1),
    drift_stream=SEAGenerator(balance_classes=False, classification_function=3, random_state=112, noise_percentage=0.1),
    position=100000,
    width=1,
    random_state=0)

# Drift @ 100.000
alternate1 = ConceptDriftStream(
    stream=SEAGenerator(balance_classes=False, classification_function=1, random_state=112, noise_percentage=0.1),
    drift_stream=alternate2,
    position=50000,
    width=1,
    random_state=0)

# Drift @ 50.000
stream = ConceptDriftStream(
    stream=SEAGenerator(balance_classes=False, classification_function=0, random_state=112, noise_percentage=0.1),
    drift_stream=alternate1,
    position=50000,
    width=1,
    random_state=0)

x, y = stream.next_sample(250000)

# Normalize and save data
save_to_file(x, y, 'sea')