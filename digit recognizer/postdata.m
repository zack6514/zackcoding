% caffe toolbox, the postprocessing of the data 
clear;clc;
feature = load('feature0.txt');
feature = feature';
[~,test_y] = max(feature);
[M,N] = size(test_y);
test_y = test_y - repmat([1], M, N);
test_y = test_y';
M = [(1:length(test_y))' test_y(:)];  
csvwrite('test_y3.csv', M);