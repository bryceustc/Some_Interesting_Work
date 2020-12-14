clc;
clear all;

% 将 CIFAR-10 数据下载到data目录 （此数据集包含 50,000 个训练图像，将用于训练 CNN。）
cifar10Data = 'data';
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
helperCIFAR10Data.download(url,cifar10Data);
% 加载训练和测试数据
[trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load(cifar10Data);
% 查看每个图像 32*32 RGB图像， 共有50000个训练样本
size(trainingImages)

% 此数据集有 10 个图像类别。可以查看具体图像类别
numImageCategories = 10;
categories(trainingLabels)

% 查看一些训练的图像
figure
thumbnails = trainingImages(:,:,:,1:100);
montage(thumbnails)

% 图像输入层 imageInputLayer，用于定义CNN可以处理数据的类型和大小
% CIFAR-10 数据 是 32 * 32 RGB的图像， 也就是32x32x3.
% ~ 表示忽略输出参数，只取前三个参数
[height,width,numChannels, ~] = size(trainingImages);
imageSize = [height width numChannels];

% 调用Deep Learning Toolbox 中的 imageInputLayer
inputLayer = imageInputLayer(imageSize)

% 定义中间层，包含多个卷积层，ReLU层和池化层组成的重复块。
% 卷积层的参数
filterSize = [5 5];
numFilters = 32;

% 中间层的具体实现
middleLayers = [

% 第一层卷积层 有32个5x5x3滤波器组，设置 padding 填充像素数为2
convolution2dLayer(filterSize,numFilters,'Padding',2)

% 下一步是添加 ReLU 非线性单元
reluLayer()

% 然后是最大池化层，池化窗口大小设置为 3，步长设置为 2
maxPooling2dLayer(3,'Stride',2)

% 重复上述三个单元块
convolution2dLayer(filterSize,numFilters,'Padding',2)
reluLayer()
maxPooling2dLayer(3, 'Stride',2)
% 重复上述三个单元块
convolution2dLayer(filterSize,2 * numFilters,'Padding',2)
reluLayer()
maxPooling2dLayer(3,'Stride',2)

]

% CNN 的最终层包括全连接层和 softmax 损失层。
finalLayers = [
    
% 添加具有64个输出神经元的完全连接层。 
fullyConnectedLayer(64)

% 添加 ReLU 非线性单元
reluLayer

% 添加最后一个完全连接的层，输出维度设置为10
fullyConnectedLayer(numImageCategories)

% 添加softmax损失层和分类层
softmaxLayer
classificationLayer
]

% 对输入层、中间层和最终层进行合并。
layers = [
    inputLayer
    middleLayers
    finalLayers
    ]

% 使用标准差为 0.0001 的正态分布随机数初始化第一个卷积层的权重。帮助训练的快速收敛。
layers(2).Weights = 0.0001 * randn([filterSize numChannels numFilters]);

% 设置网络训练算法参数
% 网络训练算法使用具有动量的随机梯度下降 SGDM
% 初始学习率为 0.001
% 在训练期间，初始学习率每 8 轮降低一次（1 轮定义为对整个训练数据集进行一次完整遍历）。训练算法运行 40 轮。
% 训练算法使用包含 128 个图像的小批量batch_size
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 128, ...
    'Verbose', true);

% 训练过程一般 需要 20-30 分钟才能完成。
% 可以加载预训练网络模型。如果要自己训练网络，请将下列代码中的 doTraining 变量设置为 true。
doTraining = false;

if doTraining    
    % 训练CNN网络.
    cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);
else
    % 加载预训练模型.
    load('rcnnStopSigns.mat','cifar10Net')       
end

% 在测试数据集上验证训练结果.
YTest = classify(cifar10Net, testImages);
accuracy = sum(YTest == testLabels)/numel(testLabels)

% 使用迁移学习方法来微调网络，以便检测停车标志
% 加载停车标志的真实值数据
data = load('stopSignsAndCars.mat', 'stopSignsAndCars');
stopSignsAndCars = data.stopSignsAndCars;

% 更新图像文件的路径匹配本地文件系统
visiondata = fullfile(toolboxdir('vision'),'visiondata');
stopSignsAndCars.imageFilename = fullfile(visiondata, stopSignsAndCars.imageFilename);
summary(stopSignsAndCars)

% 加载的训练数据包含停车标志、车头和车尾的图像文件名和关注区域标签
% 每个关注区域标签是图像内关注对象周围的边界框。训练停车标志检测器时，只需要停车标志的关注区域标签。
% 需删除车头和车尾的关注区域标签
stopSigns = stopSignsAndCars(:, {'imageFilename','stopSign'});

% 显示一张训练图像
I = imread(stopSigns.imageFilename{1});
I = insertObjectAnnotation(I,'Rectangle',stopSigns.stopSign{1},'stop sign','LineWidth',8);
figure
imshow(I)

% PS：英文停车标志数据集中只有 41 个训练图像，数据量过少，使用迁移学习
% 通过微调已基于CIFAR-10较大的数据集（有 50000 个训练图像）预训练的网络来训练学习的


% 训练R-CNN参数设置
doTraining = false;

if doTraining
    
    % 设置训练参数
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs', 100, ...
        'Verbose', true);
    
    % 训练R-CNN  
    rcnn = trainRCNNObjectDetector(stopSigns, cifar10Net, options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1])
else
    % 加载预训练模型.
    load('rcnnStopSigns.mat','rcnn')       
end

% 读取测试图像
testImage1 = imread('stopSignTest1.jpg');
testImage2 = imread('stopSignTest2.jpg');
testImage3 = imread('stopSignTest3.jpg');
testImage4 = imread('stopSignTest4.jpg');
testImage5 = imread('stopSignTest5.jpg');
testImage6 = imread('stopSignTest6.jpg');

ShowResults(rcnn, testImage1);
ShowResults(rcnn, testImage2);
ShowResults(rcnn, testImage3);
ShowResults(rcnn, testImage4);
ShowResults(rcnn, testImage5);
ShowResults(rcnn, testImage6);
