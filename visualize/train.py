from resnet import ResNet, GAP
from chainer import cuda, Variable, serializers, optimizers
import chainer.links as L
import numpy as np
import os
from progressbar import ProgressBar
import argparse
import time
import logging
import cv2


# cmd options
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--opt', default='MomentumSGD', type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--batchsize', default=8, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--initmodel', default='', type=str)
parser.add_argument('--restart', default='', type=str)
parser.add_argument('--snapshot', default=2500, type=int)
parser.add_argument('--mean', default='data/mean.npy', type=str)
args = parser.parse_args()

# load train image file list
f = open('data/train.csv')
tmp = f.read().split('\n')[:-1]
f.close()
train_list = []
for t in tmp:
    train_list.append([t.split(',')[0], int(t.split(',')[1])])

# load val image file list
f = open('data/val.csv')
tmp = f.read().split('\n')[:-1]
f.close()
val_list = []
for t in tmp:
    val_list.append([t.split(',')[0], int(t.split(',')[1])])


# load model
model = ResNet()
model.train = False
serializers.load_hdf5(args.initmodel, model)
gap = L.Classifier(GAP())
if args.restart:
    serializers.load_hdf5(args.restart, gap)
cuda.get_device(args.gpu).use()
model.to_gpu()
gap.to_gpu()


# set optimizer
optimizer = getattr(optimizers, args.opt)(args.lr)
optimizer.setup(gap)

# load mean file
mean = np.load(args.mean)
mean = cv2.resize(mean, (224, 224))

# create minibatch
def minibatch(args, num, train=True):
    data_list = train_list if train else val_list
    x = np.empty((args.batchsize, 3, 224, 224), dtype=np.float32)
    t = np.empty(args.batchsize, dtype=np.int32)
    for i, n in enumerate(num):
        data = cv2.imread(data_list[n][0])
        data = cv2.resize(data, (224, 224)).astype(np.float32) - mean
        x[i] = data.transpose(2, 0, 1)
        t[i] = data_list[n][1]
    x = Variable(cuda.cupy.asarray(x[:len(num)]), volatile = 'off' if train else 'on')
    t = Variable(cuda.cupy.asarray(t[:len(num)]), volatile = 'off' if train else 'on')
    return x, t

# create result_dir
if not os.path.exists('results'):
    os.mkdir('results')
result_dir = 'results/{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S'))
os.mkdir(result_dir)

# initial logging
f = open(os.path.join(result_dir, 'log.csv'), 'w')
f.write('epoch,iteration,loss,error\n')
f.close()
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename=os.path.join(result_dir, 'log.txt'), level=logging.DEBUG)
logging.info(args)

# initialize
ite = 0
sum_loss = 0
sum_accuracy = 0
load_num = 0

# train loop
for epoch in range(1, args.epoch+1):
    print 'epoch {}'.format(epoch)
    perm = np.random.permutation(len(train_list))
    pbar = ProgressBar(len(range(0, len(train_list), args.batchsize)))

    # one epoch
    for i in range(0, len(train_list), args.batchsize):
        x, t = minibatch(args, perm[i:i+args.batchsize])
        x = model(x)
        optimizer.update(gap, x, t)
        sum_loss += float(gap.loss.data) * len(t.data)
        sum_accuracy += float(gap.accuracy.data) * len(t.data)
        ite += 1
        load_num += len(t.data)
        pbar.update(i / args.batchsize + 1)

         # logging
        if ite % args.snapshot is 0:
            print '{} iteration, loss = {}, error = {}'.format(ite, sum_loss / load_num, 1 - sum_accuracy / load_num)
            f = open(os.path.join(result_dir, 'log.csv'), 'a')
            f.write('{},{},{},{}\n'.format(epoch, ite, sum_loss / load_num, 1 - sum_accuracy / load_num))
            f.close()
            sum_loss = 0
            sum_accuracy = 0
            load_num = 0
            serializers.save_hdf5(os.path.join(result_dir, 'model'), gap)
            serializers.save_hdf5(os.path.join(result_dir, 'state'), optimizer)
