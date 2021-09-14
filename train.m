clear all
close all
clc

%% Initialize variables
% 原始路径换掉
% trainset_path = '../data/ModelNet40/train/'; 
% testset_path  = '../data/ModelNet40/test/';
trainset_path = './3DmFV_Data/data/ModelNet40/train/';
testset_path = './3DmFV_Data/data/ModelNet40/test/';

% 原始路径换掉
% log_dir = './log/';
log_dir = './3DmFV_Data/log/';

if ~exist(log_dir,'dir')
    mkdir(log_dir);
end

%GMM variables： GMM变量
n_gaussians = 8;
n_points = 2048;
variance = (1/n_gaussians)^2;
normalize = true;
flatten = false;
inputSize = [n_gaussians, n_gaussians, n_gaussians, 20];  %8*8*8  20
%Training variables ：训练变量
numClasses = 40;
max_epoch = 300;
augmentations = [false, true, true, true, false]; %rotate, scale, translation, jitter, outliers

MiniBatchSize = 128;
ExecutionEnvironment = 'gpu'; %这里指定使用GPU
optimizer = 'adam';
InitialLearnRate = 0.001;
LearnRateSchedule = 'piecewise';
LearnRateDropPeriod = 15;
LearnRateDropFactor = 0.7;
DispatchInBackground = true;
CheckpointPath = [log_dir,'g',num2str(n_gaussians),'_n',num2str(n_points),'/'];
if ~exist(CheckpointPath, 'dir')
    mkdir(CheckpointPath)
end

%% set up the data 数据设置
[GMM] = get_3d_grid_gmm(n_gaussians, variance); %计算给定点云的3DFisher向量
[train_pc_ds] = pc_3dmfv_data_store(trainset_path, n_points, GMM, normalize, flatten, true, augmentations); %输出：一个3dmfv表示的图像数据存储对象
[test_pc_ds] = pc_3dmfv_data_store(testset_path, n_points, GMM, normalize, flatten, false, augmentations);
num_train_examples = length(train_pc_ds.Files);
ValidationFrequency = uint64(5*num_train_examples/MiniBatchSize); %validate every 5 epochs 每5个周期验证一次
%fv_train = readimage(train_pc_ds,1);
%disp('DONE');

%% set up the network and train 设置网络和训练 
 lgraph = net_3DmFV(inputSize, numClasses); %定义好3DmFV的网络层
 
 options = trainingOptions(optimizer, ...
    'MaxEpochs',max_epoch, ...
    'ValidationData',test_pc_ds, ...
    'ValidationFrequency',ValidationFrequency, ...
    'Verbose',false, ...
    'MiniBatchSize', MiniBatchSize,...
    'ExecutionEnvironment',ExecutionEnvironment,...
    'InitialLearnRate', InitialLearnRate,...
    'LearnRateSchedule', LearnRateSchedule,...
    'LearnRateDropPeriod', LearnRateDropPeriod,...
    'LearnRateDropFactor', LearnRateDropFactor,...
    'DispatchInBackground',DispatchInBackground,...
    'Shuffle', 'every-epoch',...
    'CheckpointPath',CheckpointPath,...
    'Plots','training-progress');
 
net = trainNetwork(train_pc_ds, lgraph, options); %训练出一个网络
save([CheckpointPath, '/LYY3DmFV_Net.mat'],'net', 'GMM', 'options', 'lgraph', 'augmentations', 'normalize', 'flatten', 'n_points'); % s获得训练模型和训练变量
%% test the network performance 测试网络的性能
YPred = classify(net, test_pc_ds);
YValidation = test_pc_ds.Labels;
accuracy = mean(YPred == YValidation)

