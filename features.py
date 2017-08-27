import tensorflow as tf
from scipy import ndimage
from core.vggnet import Vgg19
from core.utils import *
import numpy as np
import os
import hickle


def comp(x, y):
    x_num = int(x[12:-4])
    y_num = int(y[12:-4])
    if x_num > y_num:
        return 1
    if x_num < y_num:
        return -1
    if x_num == y_num:
        return 0


def main():
    type = ['train', 'val', 'test']
    PATH = os.getcwd()
    vgg_model_path = PATH + '/data/imagenet-vgg-verydeep-19.mat'
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        video_frames_size = 30
        for each in type:
            path = PATH + '/data/' + each + '/'
            save_path_feats = path + 'features_' + each + '.hkl'
            save_path_labels_all = path + 'labels_all_' + each + '.pkl'
            video_filename = load_pickle(path + 'video_filenames_' + each + '.pkl')
            video_filename = video_filename[0:-1] # to select certain number of videos
            labels = load_pickle(path + 'labels_' + each + '.pkl')

            # gather the whole data in the current type
            all_feats = np.ndarray([len(video_filename), video_frames_size, 196, 512], dtype=np.float32)
            all_labels = [None] * len(video_filename)

            for idx, vf in enumerate(video_filename):
                if len(list(os.walk(vf))[0][-1] ) > 10: #only read valid video
                    images_list = sorted(list(os.walk(vf))[0][-1], cmp=comp)
                    cur_images_path = [vf + '/' + image for image in images_list]
                    cur_labels = labels[idx]
                    n_examples = len(cur_images_path)
                    n_rest = video_frames_size - n_examples

                    image_batch = np.array(
                        map(lambda x: ndimage.imread(x, mode='RGB'), cur_images_path)).astype(np.float32)
                    feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})

                    zero_fill = np.zeros([n_rest, feats[0].shape[0], feats[0].shape[1]])
                    all_feats[idx, :] = np.concatenate((feats, zero_fill))
                    all_labels[idx] = [cur_labels] * n_examples + ['0'] * n_rest
                    print ('Processed' + str(idx + 1) + 'videos..')

            print 'strat to save feats and labels.'
            # use hickle to save huge feature vectors
            hickle.dump(all_feats, save_path_feats)
            save_pickle(all_labels,save_path_labels_all)
            print ("Saved %s.." % save_path_feats)

main()
