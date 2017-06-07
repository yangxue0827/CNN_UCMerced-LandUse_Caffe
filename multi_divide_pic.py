import os
import shutil
import random
import stat
import time
import multiprocessing


def build_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def divide_pic(path, _dir, divide_rate):

    print 'Dividing data-', _dir

    build_path(os.path.join(path['training_data_path'], _dir))
    build_path(os.path.join(path['testing_data_path'], _dir))

    pics = os.listdir(os.path.join(path['input_data_path'], _dir))

    num = len(pics)

    
    random.shuffle(pics)

    
    train_pic = pics[:int(divide_rate * num)]
    test_pic = pics[int(divide_rate * num):]

    curr_path = os.path.join(path['input_data_path'], _dir)

    for pic in train_pic:
        shutil.copy(os.path.join(curr_path, pic), os.path.join(path['training_data_path'], _dir, pic))
        
        os.chmod(os.path.join(path['training_data_path'], _dir, pic), stat.S_IWRITE)

    for pic in test_pic:
        shutil.copy(os.path.join(curr_path, pic), os.path.join(path['testing_data_path'], _dir, pic))
        os.chmod(os.path.join(path['testing_data_path'], _dir, pic), stat.S_IWRITE)

    print 'Dividing data-', _dir, '--done'


def multi_divide_pic(input_data_path, training_data_path, testing_data_path, divide_rate):

    start = time.time()

    
    path = dict()
    path['input_data_path'] = input_data_path
    path['training_data_path'] = training_data_path
    path['testing_data_path'] = testing_data_path

    build_path(path['training_data_path'])
    build_path(path['testing_data_path'])

    pool = multiprocessing.Pool(processes=1)

    for _dir in os.listdir(path['input_data_path']):

        pool.apply_async(divide_pic, args=(path, _dir, divide_rate))

    pool.close()
    pool.join()

    end = time.time()

    print 'Data divide complete!'
    print 'Data divide task runs ', (end - start), 's'


if __name__ == '__main__':

    input_data_path = '/home/yangxue/yangxue/UCMerced_LandUse/Temp'
    training_data_path = '/home/yangxue/yangxue/UCMerced_LandUse/Train'
    testing_data_path = '/home/yangxue/yangxue/UCMerced_LandUse/Valid'

    divide_rate = 7./8

    start = time.time()

    multi_divide_pic(input_data_path, training_data_path, testing_data_path, divide_rate)

    end = time.time()

    print 'Multi-Task runs %f' % (end - start), 's'
