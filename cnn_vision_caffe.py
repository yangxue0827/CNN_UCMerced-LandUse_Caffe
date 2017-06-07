import uuid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import caffe
import time
import sys
from collections import defaultdict
import codecs

plt.rcParams['font.sans-serif'] = ['SimHei']
# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)  # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'
# use grayscale output rather than a (potentially misleading) color heatmap
label_size = 18
plt.rcParams['xtick.labelsize'] = label_size
plt.rcParams['ytick.labelsize'] = label_size
NUM_STYLE_LABELS = 21

caffe_root = '/home/yangxue0827/caffe'
sys.path.insert(0, caffe_root + 'python')

caffe.set_mode_cpu()

print 'load the structure of the model...'
model_def = caffe_root + '/models/bvlc_reference_caffenet/deploy.prototxt'
print 'load the weights of the model...'
model_weights = caffe_root + '/models/bvlc_reference_caffenet/caffenet_train_iter_285.caffemodel'

print 'build the trained net...'
net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

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


def style_labels_dict(image_path):
    images = list(np.loadtxt(image_path, str, delimiter='\n'))
    style_label = dict()
    for image in images:
        image_list = image.split(' ')
        label = image_list[-1]
        labels_name = image_list[-2].split('/')[-2]
        if label not in style_label.keys():
            style_label[label] = labels_name
    return style_label

print 'make label_name list...'
image_root = '/home/yangxue0827/yangxue/UCMerced_LandUse'
test_image_root = image_root + '/Test_Linux/'
test_image_path = caffe_root + '/data/mydata/val.txt'
style_labels = style_labels_dict(test_image_path)


def disp_preds(image, k=5):
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['prob'][0]
    # print probs
    top_k = (-probs).argsort()[:k]
    probs_k = []
    lables_k = []
    for i, p in enumerate(top_k):
        probs_k.append(100 * probs[p])
        lables_k.append(style_labels[str(p)])

    return probs_k, lables_k


def disp_style_preds(image):
    probs_k, lables_k = disp_preds(image)
    return probs_k, lables_k


def show_acc_preclass():
    images = list(np.loadtxt(test_image_path, str, delimiter='\n'))
    preclass_num = {}
    preclass_corrct_num = {}
    preimage_error_probs = []
    preimage_error_labels = []
    preimage_error_name = []
    for label_name in style_labels.values():
        preclass_num[label_name] = 0
        preclass_corrct_num[label_name] = 0

    for i, image_path in enumerate(images):

        true_label_name = image_path.split(' ')[-2].split('/')[-2]
        preclass_num[true_label_name] = preclass_num[true_label_name] + 1

        image = caffe.io.load_image(test_image_root + image_path.split(' ')[-2])
        transformed_image = transformer.preprocess('data', image)
        probs, lables = disp_style_preds(transformed_image)
        print 'predict image ' + str(image_path.split(' ')[-2].split('/')[-1]) + ' --done'

        if true_label_name == lables[0]:
            preclass_corrct_num[true_label_name] += 1
        else:
            preimage_error_probs.append(probs)
            preimage_error_labels.append(lables)
            preimage_error_name.append(image_path.split(' ')[-2].split('/')[-1])

    # np.save(image_root + '/preimage_error_probs.npy', preimage_error_probs)
    # np.save(image_root + '/preimage_error_labels.npy', preimage_error_labels)
    # np.save(image_root + '/preimage_error_name.npy', preimage_error_name)
    f1 = codecs.open(image_root + '/caffnet/preimage_error_probs.txt', 'a+', 'utf-8')
    f1.write(str(preimage_error_probs))
    f1.close()

    f2 = codecs.open(image_root + '/caffenet/preimage_error_labels.txt', 'a+', 'utf-8')
    f2.write(str(preimage_error_labels))
    f2.close()

    f3 = codecs.open(image_root + '/caffenet/preimage_error_name.txt', 'a+', 'utf-8')
    f3.write(str(preimage_error_name))
    f3.close()

    preclass_acc = {}
    for label_name in style_labels.values():
        preclass_acc[label_name] = float(preclass_corrct_num[label_name]) / float(preclass_num[label_name])
    print 'classes testing accuracy', preclass_acc

    k = 1
    plt.figure(figsize=(8, 7))
    # plt.tight_layout()
    ind = np.arange(0, k * len(preclass_acc), k)

    colors = ['#edf8fb', '#ccece6', '#99d8c9', '#66c2a4', '#41ae76', '#238b45', '#005824', '#f1eef6', '#d4b9da',
              '#c994c7', '#df65b0', '#e7298a', '#ce1256', '#91003f', '#ffffb2', '#fed976', '#feb24c', '#fd8d3c',
              '#fc4e2a', '#e31a1c', '#b10026']
    rects = plt.bar(ind, preclass_acc.values(), width=0.7, color=colors)
    plt.xticks(ind + 0.35, preclass_acc.keys(), rotation='vertical')
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                 '%.2f' % float(height),
                 ha='center', va='bottom')
    plt.xlim([0, ind.size])
    plt.tight_layout()
    plt.savefig(image_root + '/caffenet/classes_test_accuracy.png')
    plt.show()


def vis_square(data, title):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    # print  data.shape

    plt.imshow(data)
    plt.title(title)
    plt.axis('off')
    plt.savefig(image_root + '/' + title + '.png')
    plt.show()


def vis_show():
    image_path = image_root + '/Test_Linux/airplane/airplane88_resize_gray.tif'

    image = caffe.io.load_image(image_path)
    transformed_image = transformer.preprocess('data', image)
    disp_style_preds(transformed_image)
    print 'conv1 filter output'

    filters = net.params['conv1'][0].data
    vis_square(filters.transpose(0, 2, 3, 1), 'conv1_filters')

    print 'original picture'

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    print 'conv1 output'
    feat = net.blobs['conv1'].data[0, :36]
    vis_square(feat, 'conv1_output')
    print 'conv2 output'
    feat = net.blobs['conv2'].data[0, :36]
    vis_square(feat, 'conv2_output')
    print 'conv3 output'
    feat = net.blobs['conv3'].data[0, :36]
    vis_square(feat, 'conv3_output')
    print 'conv4 output'
    feat = net.blobs['conv4'].data[0, :36]
    vis_square(feat, 'conv4_output')
    print 'conv5 pool output'
    feat = net.blobs['pool5'].data[0, :36]
    vis_square(feat, 'pool5_output')


def show_labes(image, probs, lables, true_label):
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3)
    ax1 = plt.subplot(gs[1])
    x = list(reversed(lables))
    y = list(reversed(probs))
    colors = ['#edf8fb', '#ccece6', '#99d8c9', '#66c2a4', '#41ae76']
    # colors = ['#624ea7', 'g', 'yellow', 'k', 'maroon']
    # colors=list(reversed(colors))
    width = 0.4  # the width of the bars
    ind = np.arange(len(y))  # the x locations for the groups
    ax1.barh(ind, y, width, align='center', color=colors)
    ax1.set_yticks(ind + width / 2)
    ax1.set_yticklabels(x, minor=False)
    for i, v in enumerate(y):
        ax1.text(v + 1, i, '%5.2f%%' % v, fontsize=14)
    plt.title('Probability Output', fontsize=20)
    ax2 = plt.subplot(gs[2])
    ax2.axis('off')
    ax2.imshow(image)
    plt.title(true_label, fontsize=20)
    plt.show()
    # if true_label != lables[0]:
    #     unique_filename = uuid.uuid4()
    #     fig.savefig('predit_worng/' + str(unique_filename) + '.jpg')


def show_predict():
    # images = list(np.loadtxt(test_image_path, str, delimiter='\n'))
    images = ['denseresidential/denseresidential47_resize_gray.tif 11',
              'mobilehomepark/mobilehomepark14_resize_gray.tif 13']
    for image_path in images:
        true_label = image_path.split(' ')[-2].split('/')[-2]
        image = caffe.io.load_image(test_image_root + image_path.split(' ')[-2])
        transformed_image = transformer.preprocess('data', image)
        probs, lables = disp_style_preds(transformed_image)
        show_labes(image, probs, lables, true_label)


# start = time.time()
# show_acc_preclass()
# end = time.time()
# print 'preclass predict runs %0.5f seconds' % (end - start)

start = time.time()
show_predict()
end = time.time()
print 'vision runs %0.5f seconds' % (end - start)
