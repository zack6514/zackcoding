% use the deeplearnToolbox to solve the digit recongnizer in kaggle!
clear;clc
trainFile = 'train.csv';
testFile = 'test.csv';
fidId = fopen(trainFile);

M = csvread(trainFile, 1);   % 读取csv文件除第一行以外的所有数据
train_x = M(:, 2:end);    %第2列开始为数据data
label = M(:,1)';  %第一列为标签
label(label == 0) = 10;   % 不变为10 下面一句无法处理
train_y = full(sparse(label, 1:size(train_x, 1), 1));   %将标签变成一个矩阵

train_x = double(reshape(train_x',28,28,size(train_x, 1)))/255;  



fidId = fopen('test.csv');     %% 处理预测的数据
M = csvread(testFile, 1);   % 读取csv文件除第一行以外的所有数据
test_x = double(reshape(M',28,28,size(M, 1)))/255;  
clear fidId label testFile M testFile trainFile


addpath D:\DeepLearning\DeepLearnToolbox-master\data\      %路径需要改下
addpath D:\DeepLearning\DeepLearnToolbox-master\CNN\
addpath D:\DeepLearning\DeepLearnToolbox-master\util\

rand('state',0)
cnn.layers = {        %%% 设置各层feature maps个数及卷积模板大小等属性
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};

opts.alpha = 0.01;   %迭代下降的速率
opts.batchsize = 50;   %每次选择50个样本进行更新  随机梯度下降，每次只选用50个样本进行更新
opts.numepochs = 25;   %迭代次数
cnn = cnnsetup(cnn, train_x, train_y);      %对各层参数进行初始化 包括权重和偏置
cnn = cnntrain(cnn, train_x, train_y, opts);  %训练的过程，包括bp算法及迭代过程

test_y = cnntest(cnn, test_x);      %对测试数据集进行测试
test_y(test_y == 10) = 0;      %标签10 需要反转为0
test_y = test_y';
M = [(1:length(test_y))' test_y(:)];  
csvwrite('test_y.csv', M);
figure; plot(cnn.rL);
