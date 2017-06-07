# -*- coding: utf-8 -*-
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
palette = np.array(sns.color_palette("hls", 21))
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import itertools
import matplotlib
from sklearn.utils import shuffle

caffe_root = '/home/yangxue0827/caffe/'
image_root = '/home/yangxue0827/yangxue/UCMerced_LandUse'
X_train = np.load(image_root + '/train_features11.npy')
y_train = np.load(image_root + '/train_labels11.npy')
X_test = np.load(image_root + '/test_features1.npy')
y_test = np.load(image_root + '/test_labels1.npy')
X, Y = [], []
x1, y1 = X_train.tolist(), y_train.tolist()
x2, y2 = X_test.tolist(), y_test.tolist()
X.extend(x1)
X.extend(x2)
Y.extend(y1)
Y.extend(y2)
X, Y = np.array(X), np.array(Y)


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

test_image_root = image_root + '/Test_Data/'
test_image_path = caffe_root + '/data/mydata/val1.txt'
style_labels = style_labels_dict(test_image_path)

X = X.reshape((-1, 4096))

X, Y = shuffle(X, Y, random_state=0)

# print X.shape, Y.shape
X_tsne = TSNE(n_components=2, early_exaggeration=10.0, random_state=20160530).fit_transform(X)
# X_tsne = PCA().fit_transform(X)

# fig=plt.figure()
# ax=Axes3D(fig)

markers =matplotlib.markers.MarkerStyle.filled_markers

markers = marker = itertools.cycle(markers)

f = plt.figure(figsize=(16, 8))

ax = plt.subplot(aspect='equal')

for i in range(21):
    ax.scatter(X_tsne[Y == i, 0], X_tsne[Y == i, 1], marker=markers.next(), c=palette[i], label=style_labels[str(i)])
plt.legend(loc=2, numpoints=1, ncol=2, fontsize=12, bbox_to_anchor=(1.05, 0.8))
plt.title('classes distribution')
ax.axis('off')
plt.savefig(image_root + '/t_sne.png')
plt.show()
