import os
import math
import numpy as np
import cv2
from core.utils import *
from PIL import Image


# cut the images from the video
def Video2Images(rpath, spath, name, n):
    if not os.path.exists(spath):
        os.makedirs(spath)
    m = 0
    vc = cv2.VideoCapture(rpath)
    while 1:
        rval, frame = vc.read()
        if rval:
            image_name = spath + name + '_' + str(m) + '.jpg'
            image_resized = cv2.resize(frame,(224,224),interpolation= cv2.INTER_AREA)
            cv2.imwrite(image_name , image_resized)
            m += 1
    vc.release()
    return 1


def mix_up(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    n = len(x)
    rand_idxs = np.random.permutation(n)
    x = x[rand_idxs]
    y = y[rand_idxs]
    z = z[rand_idxs]
    return x, y, z


# seprate the whole data into train, val, and test set
def seprate_data(folders):

    attributes = ['videos_id', 'labels', 'videos_filename']
    sets = ['train', 'val', 'test']

    #build necessary variables
    for attr in attributes:
        for s in sets:
            exec (attr + '_' + s + '=[]')


    for each in folders:
        path = '/home/hong/Downloads/action-recognition' + '/image/' + each + '/'
        video_id = load_pickle(path + each + '_video_ids.pkl')
        video_id = np.array(video_id)
        N = len(video_id)

        label = load_pickle(path + each + '_label.pkl')
        labels = [label] * N
        videos_filename = [path + per + '/' for per in video_id]
        VideoId = [each + '_' + id for id in video_id]

        # mix up the data within the folder
        VideoId, labels, videos_filename = mix_up(VideoId, labels, videos_filename)
        VideoId = list(VideoId)
        labels = list(labels)
        videos_filename = list(videos_filename)

        # 0.6:0.20:0.20 distributed on the sets including train, val, and test sets
        n1 = int(math.ceil(N * 0.6))
        n2 = int(math.ceil(N * 0.20))

        videos_id_train += VideoId[:n1]
        videos_id_val += VideoId[n1:n1 + n2]
        videos_id_test += VideoId[n1 + n2:]

        labels_train += labels[:n1]
        labels_val += labels[n1:n1 + n2]
        labels_test += labels[n1 + n2:]

        videos_filename_train += videos_filename[:n1]
        videos_filename_val += videos_filename[n1:n1 + n2]
        videos_filename_test += videos_filename[n1 + n2:]

    videos_id_train, labels_train, videos_filename_train = mix_up(videos_id_train, labels_train, videos_filename_train)
    videos_id_val, labels_val, videos_filename_val = mix_up(videos_id_val, labels_val, videos_filename_val)
    videos_id_test, labels_test, videos_filename_test = mix_up(videos_id_test, labels_test, videos_filename_test)

    path = '/home/hong/Downloads/action-recognition' + '/data/'

    save_pickle(videos_id_train, path + 'train/' + 'video_ids_train.pkl')
    save_pickle(videos_id_val, path + 'val/' + 'video_ids_val.pkl')
    save_pickle(videos_id_test, path + 'test/' + 'video_ids_test.pkl')

    save_pickle(labels_train, path + 'train/' + 'labels_train.pkl')
    save_pickle(labels_val, path + 'val/' + 'labels_val.pkl')
    save_pickle(labels_test, path + 'test/' + 'labels_test.pkl')

    save_pickle(videos_filename_train, path + 'train/' + 'video_filenames_train.pkl')
    save_pickle(videos_filename_val, path + 'val/' + 'video_filenames_val.pkl')
    save_pickle(videos_filename_test, path + 'test/' + 'video_filenames_test.pkl')

def main():
    folders = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a08', 'a09', 'a11', 'a12']
    open_path = '/home/hong/Downloads/action-recognition/video/multiview_action_videos/'
    path = '/home/hong/Downloads/action-recognition'
    for type in folders:
        video_path = open_path + type + '/'
        label = open(video_path + 'label' + type[1:] + '.txt').readline()[:-1]
        image_type_path = path + '/image/' + type + '/'

        # build the folder of each type -- a01, a02, ..., a12 and save the label for each type
        if not os.path.exists(image_type_path):
            os.makedirs(image_type_path)
        save_pickle(label, image_type_path + type + '_label.pkl')

        video_names = []

        images_per_video = 30
        video_txt = open(video_path + 'videos.txt').readlines()

        # cut images from videos and resize them
        for index, name in enumerate(video_txt):
            print ('video' + name[:-1] + 'process ... ')
            name = name[:-5]
            rpath = video_path + name + '.avi'
            spath = image_type_path + name + '/'
            Video2Images(rpath, spath, name, images_per_video)

            video_names.append(name)

        # save the videos_ids in image_resized folders
        save_pickle(video_names, image_type_path + '/' + type + '_video_ids.pkl')

    # divide the data into train, val,and test
    seprate_data(folders)
    # label to idx dictionary
    label_to_idx = {'pick up with one hand': 1, 'pick up with two hands': 2, 'drop trash': 3, 'walk around': 4,
                    'sit down': 5, 'stand up': 6, 'donning': 7, 'doffing': 8, 'throw': 9, 'carry': 0}
    save_pickle(label_to_idx, path + '/data/label_to_idx.pkl')

main()
