import os
import shutil
import codecs


def build_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

input_data_path = '/home/yangxue0827/yangxue/UCMerced_LandUse/Train_Data'
output_data_path = '/home/yangxue0827/yangxue/UCMerced_LandUse/'
# build_path(output_data_path)

f = codecs.open(os.path.join(output_data_path, 'train1.txt'), 'a+', 'utf-8')

label_temp = 0
for file in os.listdir(input_data_path):
    file_path = os.path.join(input_data_path, file)
    for pic in os.listdir(file_path):
        pic_path = os.path.join(file_path, pic)
        context = str(file) + '/' + str(pic) + ' ' + str(label_temp) + '\n'
        f.write(str(context))
        # shutil.copy(pic_path, output_data_path)
    label_temp += 1

f.close()
