##################################################################################
# EMD in PyTorch and Tensorflow
# This code is written by Vinit Sarode.
##################################################################################


import torch
import tensorflow as tf
import helper
import time
from emd import EMDLoss
from emd_tf import tf_util_loss

######################################### Data Loading #########################################

templates = helper.loadData('test_data')
start_idx = 0
end_idx = 32
num_point = 1024
batch_size = end_idx - start_idx


######################################### Pytorch Testing #########################################

p1 = torch.from_numpy(templates[start_idx:end_idx,0:num_point,:]).cuda()
p2 = torch.from_numpy(templates[start_idx:end_idx,0:num_point,:]).cuda()

dist =  EMDLoss()
p1.requires_grad = True
p2.requires_grad = True

print('PyTorch Results: ')
s = time.time()
cost = dist(p1, p2)
emd_time_torch = time.time() - s
loss_torch = (torch.sum(cost))/(p1.size()[1]*p1.size()[0])
print('Time: {} and Loss: {}'.format(emd_time_torch, loss_torch))


######################################### Tensorflow Testing #########################################

with tf.device('/gpu:0'):
	t1 = tf.placeholder(tf.float32, shape=(batch_size,num_point,3))
	t2 = tf.placeholder(tf.float32, shape=(batch_size,num_point,3))
	loss_tf = tf_util_loss.earth_mover(t1, t2)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Tensorflow Results: ')
start_tf = time.time()
loss_val = sess.run(loss_tf, feed_dict={t1:templates[start_idx:end_idx,0:num_point,:], t2:templates[start_idx:end_idx,0:num_point,:]})
emd_time_tf = time.time()-start_tf
print('Time: {} and Loss: {}'.format(emd_time_tf, loss_val))