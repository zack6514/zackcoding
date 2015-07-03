% use the deeplearnToolbox to solve the digit recongnizer in kaggle!
clear;clc
trainFile = 'train.csv';
testFile = 'test.csv';
fidId = fopen(trainFile);

M = csvread(trainFile, 1);   % ��ȡcsv�ļ�����һ���������������
train_x = M(:, 2:end);    %��2�п�ʼΪ����data
label = M(:,1)';  %��һ��Ϊ��ǩ
label(label == 0) = 10;   % ����Ϊ10 ����һ���޷�����
train_y = full(sparse(label, 1:size(train_x, 1), 1));   %����ǩ���һ������

train_x = double(reshape(train_x',28,28,size(train_x, 1)))/255;  



fidId = fopen('test.csv');     %% ����Ԥ�������
M = csvread(testFile, 1);   % ��ȡcsv�ļ�����һ���������������
test_x = double(reshape(M',28,28,size(M, 1)))/255;  
clear fidId label testFile M testFile trainFile


addpath D:\DeepLearning\DeepLearnToolbox-master\data\      %·����Ҫ����
addpath D:\DeepLearning\DeepLearnToolbox-master\CNN\
addpath D:\DeepLearning\DeepLearnToolbox-master\util\

rand('state',0)
cnn.layers = {        %%% ���ø���feature maps���������ģ���С������
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};

opts.alpha = 0.01;   %�����½�������
opts.batchsize = 50;   %ÿ��ѡ��50���������и���  ����ݶ��½���ÿ��ֻѡ��50���������и���
opts.numepochs = 25;   %��������
cnn = cnnsetup(cnn, train_x, train_y);      %�Ը���������г�ʼ�� ����Ȩ�غ�ƫ��
cnn = cnntrain(cnn, train_x, train_y, opts);  %ѵ���Ĺ��̣�����bp�㷨����������

test_y = cnntest(cnn, test_x);      %�Բ������ݼ����в���
test_y(test_y == 10) = 0;      %��ǩ10 ��Ҫ��תΪ0
test_y = test_y';
M = [(1:length(test_y))' test_y(:)];  
csvwrite('test_y.csv', M);
figure; plot(cnn.rL);
