# CNN_UCMerced-LandUse_Caffe（数据：http://vision.ucmerced.edu/datasets/landuse.html）
主要任务：基于深度学习框架完成对光学遥感图像UCMerced LandUse数据集的分类。 
数据特点：共包含21类土地类型图片，每类100张，每张像素大小为256*256，数据类内距离大，类间小。 
完成情况：数据量太小，训练数据出现过拟合；为了克服这个问题，又减小训练时间，采用caffe框架，在别人训练好的bvlc_reference_caffenwt模型上进行fine-tune，对最后一层设置较大的学习速率，结果取得了93%的正确率；在这基础上又在fc7层上提取了每张图片的4096维特征，进行了SVM分类，取得了95%以上的分类正确率，并对结果做了可视化分析。

环境：ubuntu14.04 + caffe + python（数据划分和增强在用windows10的3.5，其余都是unbuntu下用的2.7）
序(相关路径需要修改)/步骤：
multi_divide_pic.py---多进程进行数据划分（cv2没装成功，建议用cv2，方便）
multi_augmentation_pic.py---多进程数据增强
make_caffe_lmdb.py---生成caffe训练需要的数据路径文件，然后修改caffe配置文件
bvlc_reference_caffenet.caffemodel---caffe模型，在上面进行finetune（http://dl.caffe.berkeleyvision.org/?from=message&isappinstalled=1）
binaryproto2npy.py---将caffe生成的均值文件转换成.npy格式
cnn_vision_caffe.py---对训练好的模型进行可视化分析
extract_features.py---获取每张图片在fc7层输出的4096维特征
svm_predict.py---使用svm对上述提取的特征进行训练预测
svm_vision.py---对svm模型进行可视化分析tsne.py---对数据进行降维可视化
