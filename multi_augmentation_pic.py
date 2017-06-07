
import os
import matplotlib.image as mpimg
from scipy import misc
import shutil
import stat
import multiprocessing
import time
import numpy as np
import random


def build_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def mirror_pic(output_data_path, _dir, pic, img):

    fname, fextension = os.path.splitext(pic)

    mirror_x_img = img[:, ::-1, :]
    mirror_x_img_gray = rgb2gray(mirror_x_img)
    mirror_y_img = img[::-1, :, :]
    mirror_y_img_gray = rgb2gray(mirror_y_img)
    mirror_xy_img = img[::-1, ::-1, :]
    mirror_xy_img_gray = rgb2gray(mirror_xy_img)

    misc.imsave(os.path.join(output_data_path, _dir, (fname + '_mirror_x' + fextension)), mirror_x_img_gray)
    os.chmod(os.path.join(output_data_path, _dir, (fname + '_mirror_x' + fextension)), stat.S_IWRITE)
    misc.imsave(os.path.join(output_data_path, _dir, (fname + '_mirror_y' + fextension)), mirror_y_img_gray)
    os.chmod(os.path.join(output_data_path, _dir, (fname + '_mirror_y' + fextension)), stat.S_IWRITE)
    misc.imsave(os.path.join(output_data_path, _dir, (fname + '_mirror_xy' + fextension)), mirror_xy_img_gray)
    os.chmod(os.path.join(output_data_path, _dir, (fname + '_mirror_xy' + fextension)), stat.S_IWRITE)

    return mirror_x_img, mirror_y_img, mirror_xy_img


def cut_pic(output_data_path, _dir, pic, img, real_shape, is_train):

    shape = img.shape
    fname, fextension = os.path.splitext(pic)

    if is_train:
        # four orientations cut
        resize_img_lu = rgb2gray(img[:real_shape[0], :real_shape[1], :])
        resize_img_ru = rgb2gray(img[:real_shape[0], (shape[1] - real_shape[1]):, :])
        resize_img_ld = rgb2gray(img[(shape[0] - real_shape[0]):, :real_shape[1], :])
        resize_img_rd = rgb2gray(img[(shape[0] - real_shape[0]):, (shape[1] - real_shape[1]):, :])

        dict1 = {'lu': resize_img_lu, 'ru': resize_img_ru, 'ld': resize_img_ld, 'rd': resize_img_rd}

        for orientation in dict1:
            misc.imsave(os.path.join(output_data_path, _dir, (fname + '_resize_' + orientation + fextension)),
                        dict1[orientation])
            os.chmod(os.path.join(output_data_path, _dir, (fname + '_resize_' + orientation + fextension)),
                     stat.S_IWRITE)

    # centre_rgb
    if is_train:
        shift_x = random.randint(0, shape[1] - real_shape[1])
        shift_y = random.randint(0, shape[0] - real_shape[0])
    else:
        shift_x = (shape[1] - real_shape[1]) // 2
        shift_y = (shape[0] - real_shape[0]) // 2

    resize_img = img[shift_y:(shift_y + real_shape[0]), shift_x:(shift_x + real_shape[1]), :]
    resize_img_gray = rgb2gray(resize_img)
    misc.imsave(os.path.join(output_data_path, _dir, (fname + '_resize_gray' + fextension)), resize_img_gray)
    os.chmod(os.path.join(output_data_path, _dir, (fname + '_resize_gray' + fextension)), stat.S_IWRITE)

    if is_train:
        resize_img_r = resize_img[:, :, 0]
        resize_img_g = resize_img[:, :, 1]
        resize_img_b = resize_img[:, :, 2]

        dict2 = {'r': resize_img_r, 'g': resize_img_g, 'b': resize_img_b}

        for color in dict2:
            misc.imsave(os.path.join(output_data_path, _dir, (fname + '_resize_' + color + fextension)), dict2[color])
            os.chmod(os.path.join(output_data_path, _dir, (fname + '_resize_' + color + fextension)), stat.S_IWRITE)

    return resize_img


def augmentation_pic(real_shape, path, _dir, output_data_path, is_train):

    print('Processing-', _dir)

    for pic in os.listdir(path):
        pic_path = os.path.join(path, pic)
        img = mpimg.imread(pic_path)
        resize_img = cut_pic(output_data_path, _dir, pic, img, real_shape, is_train)

        if is_train:
            mirror_pic(output_data_path, _dir, pic, resize_img)

    print('Processing-', _dir, '--done')


def multi_augmentation_pic(input_data_path, output_data_path, shape, is_train):

    pool = multiprocessing.Pool(processes=4)

    for _dir in os.listdir(input_data_path):
        file_path = os.path.join(input_data_path, _dir)
        build_path(os.path.join(output_data_path, _dir))

        pool.apply_async(augmentation_pic, args=(shape, file_path, _dir, output_data_path, is_train))

    pool.close()
    pool.join()


if __name__ == '__main__':
    
    start = time.time()

    input_data_path = '/home/yangxue0827/yangxue/UCMerced_LandUse/Test'
    output_data_path = '/home/yangxue0827/yangxue/UCMerced_LandUse/Test_data'
    build_path(output_data_path)
    shape = [227, 227]
    multi_augmentation_pic(input_data_path, output_data_path, shape, is_train=False)
    # shutil.rmtree(input_data_path)
    end = time.time()

    print('Multi-Task runs % f' % (end - start))
