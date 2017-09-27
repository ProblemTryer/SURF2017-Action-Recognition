import numpy as np
import cPickle as pickle
import hickle
import time
import os
import collections


# # load the train, val, test data set
# def load_data(data_path='./data', split='train'):
#     # 16 images for each video
#     start_t = time.time()
#     data = {}
#
#     data['features'] = hickle.load(data_path + '/' + split + '/' + 'AAAAfeatures_' + split + '.hkl')
#     print 'end loading features.'
#     data['labels'] = hickle.load(data_path + '/' + split + '/' + 'AAAAlabels_all_' + split + '.hkl')
#     with open(data_path + '/' + split + '/' + 'AAAAlabels_all_' + split + '.pkl', 'rb') as f:
#         data['labels'] = pickle.load(f)
#     # with open(data_path + '/' + split + '/' + 'labels_' + split + '.pkl', 'rb') as f:
#     #     data['labels'] = pickle.load(f)
#     with open(data_path + '/' + split + '/' + 'video_ids_' + split + '.pkl', 'rb') as f:
#         data['video_ids'] = pickle.load(f)
#     with open(data_path + '/' + split + '/' + 'video_filenames_' + split + '.pkl', 'rb') as f:
#         data['video_filenames'] = pickle.load(f)
#
#     if split == 'train':
#         with open(os.path.join(data_path + '/' + split, 'label_to_idx.pkl'), 'rb') as f:
#             data['label_to_idx'] = pickle.load(f)
#
#     end_t = time.time()
#     print "Elapse time: %.2f" % (end_t - start_t)
#     return data


def load_data(data_path, split):
    start_t = time.time()
    features = hickle.load(data_path +  '/' + split + '/' + 'AAAAfeatures_' + split + '.hkl')
    # labels = hickle.load(data_path +  '/' + split + '/' + 'AAAAlabels_all_' + split + '.hkl')
    labels = load_pickle(data_path +  '/' + split + '/' + 'AAAAlabels_all_' + split + '.pkl')
    video_ids = load_pickle(data_path + '/' +  split + '/' + 'video_ids_' + split + '.pkl')  # name == id
    video_filename = load_pickle(data_path + '/' +  split + '/' + 'video_filenames_' + split + '.pkl')
    data = {'features': features, 'labels': labels, 'video_ids': video_ids, 'video_filenames': video_filename}
    if split == 'train':
        with open(os.path.join(data_path + '/' + split, 'label_to_idx.pkl'), 'rb') as f:
                 data['label_to_idx'] = pickle.load(f)
    end_t = time.time()
    print "Elapse time: %.2f" % (end_t - start_t)
    return data


def decode(gen_label_list, idx_to_label):
    N = len(gen_label_list[0])
    labels_video = [None] * N
    label_idxs_video = [None] * N
    for j in range(N):
        possible_results = [label_t[j] for label_t in gen_label_list]
        temp = collections.Counter(possible_results).most_common()[0][0]
        label_idxs_video[j] = temp
        labels_video[j] = idx_to_label[temp]
    return np.array(label_idxs_video), np.array(labels_video)


def accurate_percentage(x, y):
    isSame = x - y
    return float(sum([1 for each in isSame if each == 0])) / float(len(x))


def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' %path)
        return file  


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)


