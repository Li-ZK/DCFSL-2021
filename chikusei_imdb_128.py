import numpy as np
# import torch.utils.data as data
from sklearn.decomposition import PCA
import random
# import os
import pickle
import h5py
import hdf5storage
from  sklearn import preprocessing
import scipy.io as sio


def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)), 'constant', constant_values=0)
    return new_matrix

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex( assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def sampling(groundTruth):
    labels_loc = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices

    whole_indices = []
    for i in range(m):
        whole_indices += labels_loc[i]

    np.random.shuffle(whole_indices)
    return whole_indices


def load_data_HDF(image_file, label_file):
    image_data = hdf5storage.loadmat(image_file)
    label_data = hdf5storage.loadmat(label_file)
    data_all = image_data['chikusei']  # data_all:ndarray(2517,2335,128)
    label = label_data['GT'][0][0][0]  # label:(2517,2335)

    [nRow, nColumn, nBand] = data_all.shape
    print('chikusei', nRow, nColumn, nBand)
    gt = label.reshape(np.prod(label.shape[:2]), )
    del image_data
    del label_data
    del label

    data_all = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    print(data_all.shape)
    data_scaler = preprocessing.scale(data_all)
    data_scaler = data_scaler.reshape(2517,2335,128)

    return data_scaler, gt

def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key]
    label = label_data[label_key]
    gt = label.reshape(np.prod(label.shape[:2]), )

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    print(data.shape)
    data_scaler = preprocessing.scale(data)
    data_scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    return data_scaler, gt

def getDataAndLabels(trainfn1, trainfn2):
    if ('Chikusei' in trainfn1 and 'Chikusei' in trainfn2):
        Data_Band_Scaler, gt = load_data_HDF(trainfn1, trainfn2)
    else:
        Data_Band_Scaler, gt = load_data(trainfn1, trainfn2)

    del trainfn1, trainfn2
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    # SSRN
    patch_length = 4 # neighbor 9 x 9
    whole_data = Data_Band_Scaler
    padded_data = zeroPadding_3D(whole_data, patch_length)
    del Data_Band_Scaler

    np.random.seed(1334)

    whole_indices = sampling(gt)
    print('the whole indices', len(whole_indices))  # 520

    nSample = len(whole_indices)
    x = np.zeros((nSample, 2 * patch_length + 1, 2 * patch_length + 1, nBand))
    y = gt[whole_indices] - 1  # label 1-19->0-18

    whole_assign = indexToAssignment(whole_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    print('indexToAssignment is ok')
    for i in range(len(whole_assign)):
        x[i] = selectNeighboringPatch(padded_data, whole_assign[i][0], whole_assign[i][1],
                                      patch_length)
    print('selectNeighboringPatch is ok')

    print(x.shape)
    del whole_assign
    del whole_data
    del padded_data

    imdb = {}
    imdb['data'] = np.zeros([nSample, 2 * patch_length + 1, 2 * patch_length + 1, nBand], dtype=np.float32)  # <class 'tuple'>: (9, 9, 100, 77592)
    imdb['Labels'] = np.zeros([nSample], dtype=np.int64)  # <class 'tuple'>: (77592,)
    imdb['set'] = np.zeros([nSample], dtype=np.int64)

    for iSample in range(nSample):
        imdb['data'][iSample, :, :, :, ] = x[iSample, :, :, :]  # (9, 9, 100, 77592)
        imdb['Labels'][iSample] = y[iSample]  # (77592, )
        if iSample % 100 == 0:
            print('iSample', iSample)

    imdb['set'] = np.ones([nSample]).astype(np.int64)
    print('Data is OK.')

    return imdb

train_data_file = '/home/dell/lm/RN-NEW/datasets/Chikusei/HyperspecVNIR_Chikusei_20140729.mat'
train_label_file = '/home/dell/lm/RN-NEW/datasets/Chikusei/HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat'

imdb = getDataAndLabels(train_data_file, train_label_file)#14类

with open('datasets/Chikusei_imdb_128.pickle', 'wb') as handle:
    pickle.dump(imdb, handle, protocol=4)

print('Images preprocessed')