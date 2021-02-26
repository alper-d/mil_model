import os
import numpy as np
import pandas
import json
import torch
from matplotlib import pyplot
def get_min_video_size(subjects_path = ''):
  im_lenght = {}
  min_image_path = ''
  total_video_no = 0
  subjects_list = os.listdir(subjects_path)
  each_sub_vid = np.zeros(53)
  for i,subject in enumerate(subjects_list):
    # video_samples = os.listdir(os.path.join(subjects_path, subject))
    video_samples = [vid for vid in os.listdir(os.path.join(subjects_path, subject)) if os.path.isdir(os.path.join(subjects_path, subject))]
    each_sub_vid[i] =len(video_samples)
    total_video_no += len(video_samples)
    for video in video_samples:
      video_length = len(os.listdir(os.path.join(subjects_path, subject, video)))
      im_lenght["{},{}".format(str(subject), str(video))] = video_length
  print(im_lenght.values().__len__())
  hist = pyplot.hist(im_lenght.values(),300)
  pyplot.xticks(range(0,600,20))
  pyplot.show()

  with open(os.path.join(subjects_path, '..','video_len.json'), 'w+') as file:
    file.write(json.dumps(im_lenght,indent=0))

  return im_lenght, min_image_path,total_video_no,each_sub_vid

def produce_input(subjects_path = ''):
  subjects_list = os.listdir(subjects_path)
  classes = {1: 'dislike', 2: 'like', 3: 'notr'}

def split_dataset():

  x = np.arange(602)

  x = np.concatenate((np.zeros(550), np.ones(52)))
  rnd_idx = np.random.permutation(x)
  train_idx, valid_idx =rnd_idx[:550],rnd_idx[550:]
  return train_idx,valid_idx

def extract_features(path):
  path_train = os.path.join(path, 'valid_data')
  one_img_shape = (512,7,7)
  for video in os.listdir(path_train):
    vid_tensor = torch.load(os.path.join(path_train, video))
    prop_tensor = torch.empty((4, *one_img_shape))
    prop_tensor[0] = torch.mean(vid_tensor, 0)
    prop_tensor[1] = torch.std(vid_tensor, 0)
    prop_tensor[2] = torch.max(vid_tensor, 0).values
    prop_tensor[3] = torch.min(vid_tensor, 0).values
    torch.save(prop_tensor, os.path.join(path, 'valid_data_props', video))

if __name__ == '__main__':
  with open(os.path.join('..','video_len.json'),'r') as file:
    a = json.loads(file.read())

  if True:
    extract_features(os.path.join('..', 'video_tensors'))
  else:
    data_path = os.path.join(os.getcwd(), '..', '')
    data_label_path = os.path.join(os.getcwd(), '..', 'data.xlsx')

    subjects = os.listdir(data_path)
    print(os.listdir(data_path))

    label_frame = pandas.read_excel(data_label_path, sheet_name='fv-svm')
    label_frame.head(10)
    label_frame = label_frame.drop('Path', axis=1)
    a,b,c,d = get_min_video_size(data_path)
    split_dataset()
    print(a,b,c)

  print("end")