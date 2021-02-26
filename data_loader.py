import torch.utils.data as data
import torch
import os
import pandas
import numpy as np
import math
import data_util as dl
class Taste_Dataset(data.Dataset):
    def __init__(self, data_path = '', tag = '',indexes = None,  bag_size = 0, recreate = False):
        print('Dataset initialized!')
        self.subject_len = 54
        data_label_path = os.path.join(os.getcwd(), '..', 'data.xlsx')
        self.label_frame = pandas.read_excel(data_label_path, sheet_name='fv-svm')
        self.label_frame.head(10)
        self.label_frame = self.label_frame.drop('Path', axis=1)
        self.tag = tag
        if not recreate:
            self.data_path = data_path
            self.video_tensors_path = [path for path in os.listdir(os.path.join('..','video_tensors',data_path)) if path.endswith('.pt')]
            self.subject_id = list(map(lambda x:x[:4],self.video_tensors_path))
            self.video_no = list(map(lambda x:int(x[5:8]),self.video_tensors_path))
            self.is_folded = False
            if(not (type(indexes) == np.ndarray or type(indexes) == list)):
                print('a')
                self.is_folded = False
            else:
                if type(indexes) == list:
                    print('b'+ str(len(np.concatenate(indexes))))
                    self.k_fold_indexes = np.concatenate(indexes)
                else:
                    print('c'+ str(len(indexes)))
                    self.k_fold_indexes = indexes
                self.is_folded = True    
            print(self.is_folded)
            return

        total_video_no = 602
        one_image_shape = (512, 7, 7)
        sub_list = os.listdir(data_path)
        x = np.concatenate((np.zeros(550), np.ones(52)))
        rd_ind = np.random.permutation(x)

        vid_iterator = 0
        for subject in sub_list:
            for vid_index, video in enumerate(os.listdir(os.path.join(data_path, subject))):
                image_names = [f for f in os.listdir(os.path.join(data_path, subject, video)) if f.endswith('.pt')]
                img_iter = iter(image_names)
                #img_iter = ImageIterator(image_names)
                #video_bag = torch.empty((bag_size,*one_image_shape))
                bag_size = len(image_names)
                data_tensor = torch.empty((bag_size, *one_image_shape))
                for img_index in range(bag_size):
                    data_tensor[img_index, ...] = torch.from_numpy(torch.load(os.path.join(data_path, subject, video, next(img_iter).split('.')[0]+'.pt')))

                save_set = 'train_set' if rd_ind[vid_iterator] == 0 else 'valid_set'
                print('saved' + os.path.join('..', 'video_tensors',save_set,'{}_{}.pt'.format(subject,video)))
                torch.save(data_tensor, os.path.join('..', 'video_tensors',save_set,'{}_{}.pt'.format(subject,video)))
                vid_iterator += 1


    def __getitem__(self, index):
        subj = self.video_tensors_path[index][:4]
        vid = self.video_tensors_path[index][5:8]
        # print(subj, vid + '--->', os.path.join('..','video_tensors',self.data_path, self.video_tensors_path[index]))
        label = self.label_frame.loc[self.label_frame.Subject == subj].loc[self.label_frame.Video == int(vid)]['Label']
        if(self.is_folded):
            return torch.load(os.path.join('..','video_tensors',self.data_path, self.video_tensors_path[self.k_fold_indexes[index]]))[2,:], int(label)-1

        return torch.load(os.path.join('..','video_tensors',self.data_path, self.video_tensors_path[index]))[2,:], int(label)-1

    def __len__(self):
        if(self.is_folded):
            print(self.tag)
        if self.is_folded:
            # print(str(self.tag) + '-->' + str(len(self.k_fold_indexes)))
            return len(self.k_fold_indexes)
        # print(str(self.tag) + '-->' + str(len(self.video_tensors_path)))
        return len(self.video_tensors_path)


# class ImageIterator():
#     def __init__(self,myList):
#         self.iterator = iter(myList)

#     def __next__(self):
#         try:
#             return next(self.iterator)
#         except StopIteration:
#             pass

if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'bitenler')
    t = Taste_Dataset(data_path, 600, True )