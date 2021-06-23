import numpy as np

def normalize_1(X, std_old, idx_newdata):
    #update every N steps
    new_data = X[idx_newdata:]
    old_data = X[:idx_newdata]
    std = np.std(new_data,axis=0)
    if(not std_old == None):
        old_data = old_data*std_old
    X = np.concatenate((old_data,new_data), axis=0)
    X /= std

    return X 

