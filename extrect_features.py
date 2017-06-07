import numpy as np
import caffe
import sys
import time
from numba import jit
NUM_STYLE_LABELS = 21

caffe_root = '/home/yangxue0827/caffe'

image_root = '/home/yangxue0827/yangxue/UCMerced_LandUse'
train_image_root = image_root + '/Train_Linux/'
train_images_path = caffe_root + '/data/mydata/train.txt'
test_image_root = image_root + '/Test_Linux/'
test_images_path = caffe_root + '/data/mydata/val.txt'

sys.path.insert(0, caffe_root + 'python')

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + '/models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + '/models/bvlc_reference_caffenet/caffenet_train_iter_285.caffemodel',
                caffe.TEST)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + '/data/mydata/mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
# print 'mean-subtracted values:', zip('BGR', mu),mu
# create trasformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR


@jit
def show_predict(image_root, images_path):
    images = list(np.loadtxt(images_path, str, delimiter='\n'))
    features = []
    labels = []
    for image in images:
        image_list = image.split(' ')
        true_label = image_list[-1]
        image = caffe.io.load_image(image_root + image_list[-2])
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[0, ...] = transformed_image
        net.forward(start='conv1')
        feat = net.blobs['fc7'].data.copy()

        label_num = int(true_label)
        features.append(feat[0])
        labels.append(label_num)
        print 'read image ' + str(image_list[-2].split('/')[-1]) + ' --done'

    return features, labels

# start = time.time()
# test_features, test_labels = show_predict(test_image_root, test_images_path)
# np.save(image_root + '/test_features1.npy', test_features)
# np.save(image_root + '/test_labels1.npy', test_labels)
# end = time.time()
# print 'test_features extrect runs %0.5f seconds' % (end - start)

start1 = time.time()
train_features, train_labels = show_predict(train_image_root, train_images_path)
np.save(image_root + '/train_features2.npy', train_features)
np.save(image_root + '/train_labels2.npy', train_labels)
end1 = time.time()
print 'train_features extrect runs %0.5f seconds' % (end1 - start1)
