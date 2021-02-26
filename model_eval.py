from sklearn import svm
from sklearn import neighbors
import torch
import os
from os.path import join as pjoin
import numpy as np
import pandas

data_path = pjoin('..', 'video_tensors', 'train_data_props')
data_label_path = pjoin('..', 'data.xlsx')

label_frame = pandas.read_excel(data_label_path, sheet_name='fv-svm').drop('Path', axis=1)

img_shape = (512, 7, 7)
train_data = np.empty((len(os.listdir(data_path)), 512*7*7))
train_labels = np.empty(len(os.listdir(data_path)))
for i, video in enumerate(os.listdir(data_path)):
    subj = video[:4]
    vid = video[5:8]
    label = int(label_frame.loc[label_frame.Subject == subj].loc[label_frame.Video == int(vid)]['Label'])
    train_data[i,:] = torch.load(pjoin(data_path, video))[0,:].flatten().numpy()
    train_labels[i] = label

clf = neighbors.KNeighborsClassifier(n_neighbors=3, algorithm= 'brute')
clf.fit(train_data,train_labels)

test_path = pjoin('..', 'video_tensors', 'valid_data_props')
test_data = np.empty((len(os.listdir(test_path)), 512*7*7))
test_labels = np.empty(len(os.listdir(test_path)))
for i, video in enumerate(os.listdir(test_path)):
    subj = video[:4]
    vid = video[5:8]
    label = int(label_frame.loc[label_frame.Subject == subj].loc[label_frame.Video == int(vid)]['Label'])
    test_data[i,:] = torch.load(pjoin(test_path, video))[0,:].flatten().numpy()
    test_labels[i] = label

predictions = clf.predict(test_data)

conf_matrix = np.zeros((3,3))
for i,e in enumerate(predictions):
    conf_matrix[int(e-1),int(test_labels[i]-1)] += 1
print(conf_matrix)
print('end')

