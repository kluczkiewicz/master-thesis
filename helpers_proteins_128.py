import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.metrics import roc_curve


def load_data(): 
    ### TRAIN LOADING ###
    file_path_01 = "/workspace/shared/kluczkiewicz/MGR_FINAL/proteins_128/data/data_proteins_01_train.pkl"
    with open(file_path_01, 'rb') as f:
        data_01 = pickle.load(f)

    file_path_31 = "/workspace/shared/kluczkiewicz/MGR_FINAL/proteins_128/data/data_proteins_31_train.pkl"
    with open(file_path_31, 'rb') as f:
        data_31 = pickle.load(f)

    data = data_01['0_1']
    data = np.concatenate((data, data_31['3_1']))

    labels = ['0_1'] * len(data_01['0_1'])
    labels = labels + ['3_1'] * len(data_31['3_1'])
    labels = np.array(labels)

    print('Loaded data:')
    print(file_path_01)
    print(file_path_31)
    print('Final dataset created: ')
    print('Dataset shape: ', data.shape)
    print('0_1 shape: ', data_01['0_1'].shape)
    print('3_1 shape: ', data_31['3_1'].shape)

    ### TEST LOADING ###
    file_path_01_test = f"/workspace/shared/kluczkiewicz/MGR_FINAL/proteins_128/data/data_proteins_01_test.pkl"
    with open(file_path_01_test, 'rb') as f:
        data_01_test = pickle.load(f)

    file_path_31_test = f"/workspace/shared/kluczkiewicz/MGR_FINAL/proteins_128/data/data_proteins_31_test.pkl"
    with open(file_path_31_test, 'rb') as f:
        data_31_test = pickle.load(f)

    data_test = data_01_test['0_1']
    data_test = np.concatenate((data_test, data_31_test['3_1']))

    labels_test = ['0_1'] * len(data_01_test['0_1'])
    labels_test = labels_test + ['3_1'] * len(data_31_test['3_1'])
    labels_test = np.array(labels_test)

    print('Loaded data:')
    print(file_path_01_test)
    print(file_path_31_test)
    print('Final TEST dataset created: ')
    print('Dataset shape: ', data_test.shape)
    print('0_1 shape: ', data_01_test['0_1'].shape)
    print('3_1 shape: ', data_31_test['3_1'].shape)

    # knot_types = ("0_1", "3_1")

    ### SCALING TRAIN ###
    data_scaled = []

    reshaped_data = data.reshape(data.shape[0]*data.shape[1], 3)
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    scaler3 = StandardScaler()
    scaler1.fit(reshaped_data[:,0].reshape(-1, 1))
    scaler2.fit(reshaped_data[:,1].reshape(-1, 1))
    scaler3.fit(reshaped_data[:,2].reshape(-1, 1))

    for sample in data:
        scaled_sample = np.zeros((sample.shape[0], 3))
        scaled_sample[:,0] = scaler1.transform(sample[:,0].reshape(-1, 1)).reshape(-1)
        scaled_sample[:,1] = scaler2.transform(sample[:,1].reshape(-1, 1)).reshape(-1)
        scaled_sample[:,2] = scaler3.transform(sample[:,2].reshape(-1, 1)).reshape(-1)
        data_scaled.append(scaled_sample)

    data = np.array(data_scaled)

    one = OneHotEncoder()
    ya = labels.reshape(-1, 1)
    labels = one.fit_transform(ya).toarray()

    print('DATASETS SHAPES: ')
    print(data.shape, labels.shape)

    print('Example from training dataset:')
    print(data[0])
    print(labels[0])


    ### SCALING TRAIN ###
    data_scaled_test = []

    reshaped_data_test = data_test.reshape(data_test.shape[0]*data_test.shape[1], 3)
    # scaler1 = StandardScaler()
    # scaler2 = StandardScaler()
    # scaler3 = StandardScaler()
    scaler1.fit(reshaped_data_test[:,0].reshape(-1, 1))
    scaler2.fit(reshaped_data_test[:,1].reshape(-1, 1))
    scaler3.fit(reshaped_data_test[:,2].reshape(-1, 1))

    for sample in data_test:
        scaled_sample = np.zeros((sample.shape[0], 3))
        scaled_sample[:,0] = scaler1.transform(sample[:,0].reshape(-1, 1)).reshape(-1)
        scaled_sample[:,1] = scaler2.transform(sample[:,1].reshape(-1, 1)).reshape(-1)
        scaled_sample[:,2] = scaler3.transform(sample[:,2].reshape(-1, 1)).reshape(-1)
        data_scaled_test.append(scaled_sample)

    data_test = np.array(data_scaled_test)

    # one = OneHotEncoder()
    ya_test = labels_test.reshape(-1, 1)
    labels_test = one.fit_transform(ya_test).toarray()

    print('DATASETS SHAPES: ')
    print(data_test.shape, labels_test.shape)

    print('Example from training dataset:')
    print(data_test[0])
    print(labels_test[0])

    x_train = data
    y_train = labels

    x_test = data_test
    y_test = labels_test

    return x_train, y_train, x_test, y_test


def printScores(model, X, Y, roc=False):

    print("Classification report:")
    pred = model.predict(X)
    cr = classification_report(np.argmax(Y, axis=1), np.argmax(pred, axis=1))
    print(cr)

    print("Confusion matrix:")
    cm = confusion_matrix(np.argmax(Y, axis=1), np.argmax(pred, axis=1))
    print(cm)

    print('Wrong predict:')
    wp = confusion_matrix(np.argmax(Y, axis=1), np.argmax(pred, axis=1)).sum() - confusion_matrix(np.argmax(Y, axis=1), np.argmax(pred, axis = 1)).trace()
    print(wp)

    fpr, tpr, thresholds = roc_curve(np.argmax(Y, axis=1), np.array(pred)[:,1])
    roc = {'fpr': fpr, 'tpr': tpr}
    df_roc = pd.DataFrame(roc)
    df_roc.to_csv('roc.csv')

    return accuracy_score(np.argmax(Y, axis=1), np.argmax(pred, axis=1))