clear all
close all
clc

%old path
% results_path = './log/g8/';
% testset_path  = './itzik/MatlabProjects/3DmFVNet/data/ModelNet40/test/';

%new path in my computer
results_path = 'E:/LYYgithub/3DmFV_Data/log/g8_1024/';
testset_path  = 'E:/LYYgithub/3DmFV_Data/data/ModelNet40/test/';


load([results_path, '3DmFV_Net.mat']);

[test_pc_ds] = pc_3dmfv_data_store(testset_path, n_points, GMM, normalize, flatten, false, augmentations);

YPred = classify(net, test_pc_ds);
YValidation = test_pc_ds.Labels;
accuracy = mean(YPred == YValidation)