# EMD: Earth Mover Distance
It contains the Earth Mover Distance function with CUDA implementation for PyTorch as well as Tensorflow.

### Test Entire Package:
> chmod +x test.sh\
> ./test.sh

### Some Important Points about usage:
1. Both Inputs should have dimension (Batch x Number of Points x Dimension)
2. Both Inputs should be defined on GPU + CUDA devices.

## EMD with PyTorch:
Copy the \[[emd_torch](https://github.com/vinits5/emd/blob/master/emd_torch/)\] folder to desired location.

### Compile the CUDA code
> cd emd_torch\
> sudo python3 setup.py install

### Sample to Test the EMD with PyTorch
> import torch\
> from emd import EMDLoss\
> cost = EMDLoss()\
> p1, p2 = torch.rand(32,1024,3).cuda().double(), torch.rand(32,1024,3).cuda().double()\
> p1.requires_grad = True\
> p2.requires_grad = True\
> loss_val = cost(p1, p2)\
> loss_val.backward()\
> print(loss_val)\
> print(p1.grad, p2.grad)

## EMD with Tensorflow:
Copy the \[[emd_tf](https://github.com/vinits5/emd/blob/master/emd_tf/)\] folder to desired location.\
Change the import path in *tf_util_loss.py*.

### Compile the CUDA code
> cd emd_tf/pc_distance\
> make -f makefile

### Sample to Test the EMD with Tensorflow
> import tensorflow as tf\
> from emd_tf import tf_util_loss\
> loss_val = tf_util_loss.earth_mover(xyz1, xyz2)\
> loss_val = sess.run(loss, feed_dict={xyz1: xyz1_numpy, xyz2: xyz2_numpy})\
[Note: xyz1, xyz2 should be placeholders defined on GPU device]