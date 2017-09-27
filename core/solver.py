import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os 
import cPickle as pickle
from scipy import ndimage
from utils import *
from bleu import evaluate


class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17) 
                - image_idxs: Indices for mapping caption to image of shape (400000, ) 
                - word_to_idx: Mapping dictionary from word to index 
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path 
            - model_path: String; model path for saving 
            - test_model: String; model path for test 
        """

        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)


    def train(self):
        # train/val dataset

        # n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))
        features = self.data['features']
        print 'dsafgd'
        labels = np.array(self.data['labels'])
        print labels
        video_ids = np.array(self.data['video_ids'])
        n_examples = video_ids.shape[0]
        # captions = self.data['captions']
        # image_idxs = self.data['image_idxs']

        # val_features = self.val_data['features']
        # n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))

        n_iters_per_epoch = int(len(labels) / self.batch_size)

        # build graphs for training model and sampling captions
        loss = self.model.build_model()
        print 'builded'

        tf.get_variable_scope().reuse_variables()
        _, _, sampled_labels = self.model.build_sampler()

        # train op
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)  #changed
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # summary op
        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)
        for grad, var in grads_and_vars:
            tf.histogram_summary(var.op.name+'/gradient', grad)

        summary_op = tf.merge_all_summaries()

        print "The number of epoch: %d" %self.n_epochs
        print "Data size: %d" %(len(labels))
        print "Batch size: %d" %self.batch_size
        print "Iterations per epoch: %d" %n_iters_per_epoch

        config = tf.ConfigProto(allow_soft_placement = True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.initialize_all_variables().run()
            summary_writer = tf.train.SummaryWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            labels_dirt = load_pickle('/home/hong/Downloads/action-recognition/data/train/label_to_idx.pkl')
            print labels_dirt

            for e in range(self.n_epochs):
                # rand_idxs = np.random.permutation(10)
                # labels = labels[rand_idxs]
                # # print labels
                # video_ids = video_ids[rand_idxs]
                # features = features[rand_idxs]

                for i in range(n_iters_per_epoch): # or use directory
                    labels_batch = labels[i*self.batch_size:(i+1)*self.batch_size]
                    print np.shape(labels_batch)
                    video_idxs_batch = video_ids[i*self.batch_size:(i+1)*self.batch_size]
                    # label_batch_idxs = np.array([self.model.label_to_idx[per] for per in labels_batch]
                    # label_batch_idxs = []
                    # for per in labels_batch:
                    #     print self.model.label_to_idx(per)
                    # print label_batch_idxs
                    # features_batch = tf.reduce_sum(features[i*self.batch_size:(i+1)*self.batch_size],0)
                    features_batch = features[i * self.batch_size:(i + 1) * self.batch_size]
                    # for per in range(self.batch_size):
                    # print features_batch[per].dtype
                    label_batch_idxs = []
                    for PER in labels_batch:
                        label_batch_idxs.append(labels_dirt[PER[0]])

                    # bbb = labels_batch[0][0]
                    # aaa = np.array([labels_dirt[bbb]])
                    # print aaa
                    # np.shape(features_batch[per])
                    feed_dict = {self.model.features: features_batch, self.model.label_idxs: label_batch_idxs }  #long(labels_batch[per].replace('\n',''))
                    # feed_dict = feed_dict.replace('/n','')
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l
                    print '$$$$$$$$$$$$$$$$$'
                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, e*n_iters_per_epoch + i)

                    if (i+1) % self.print_every == 0:
                        print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" %(e+1, i+1, l)
                        sam_labels_list = sess.run(sampled_labels, feed_dict)


                        gen_label_idxs, gen_labels = decode(sam_labels_list, self.model.idx_to_label)
                        ground_truths = label_batch_idxs[:]

                        for j in range(len(ground_truths)):
                            print(video_idxs_batch[j])
                            Ground_truth = 'org: ' + str(ground_truths[j])
                            Generated_one = 'gen: ' + str(gen_label_idxs[j])
                            print(Ground_truth + '--V.S.--' + Generated_one)
                        print('the current accurancy rate: ' +
                              str(accurate_percentage(gen_label_idxs, ground_truths)))

                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "Elapsed time: ", time.time() - start_t
                prev_loss = curr_loss
                curr_loss = 0


                # save model's parameters
                if (e+1) % self.save_every == 0:
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e+1)
                    print "model-%s saved." %(e+1)


    def test(self, data, split='train', attention_visualization=True, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17) 
            - image_idxs: Indices for mapping caption to image of shape (24210, ) 
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        features = data['features']

        # waiting to add





