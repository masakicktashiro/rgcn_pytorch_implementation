import numpy as np

def get_splits(y, train_idx, test_idx, validation=True):
    # Make dataset splits
    # np.random.shuffle(train_idx)
    if validation:
        idx_train = train_idx[len(train_idx) / 5:] 
        idx_val = train_idx[:len(train_idx) / 5]
        idx_test = idx_val  # report final score on validation set for hyperparameter optimization
    else:
        idx_train = train_idx
        idx_val = train_idx  # no validation
        idx_test = test_idx

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)

    y_train[idx_train] = np.array(y[idx_train].todense())
    y_val[idx_val] = np.array(y[idx_val].todense())
    y_test[idx_test] = np.array(y[idx_test].todense())

    return y_train, y_val, y_test, idx_train, idx_val, idx_test
