clear all;
clc;

% 如需自己训练模型，将doTraining 变量设置为 true
doTraining = false;
if ~doTraining && ~exist('yolov2ResNet50VehicleExample_19b.mat','file')    
    disp('Downloading pretrained detector (98 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/yolov2ResNet50VehicleExample_19b.mat';
    websave('yolov2ResNet50VehicleExample_19b.mat',pretrainedURL);
end

% 加载数据集
unzip vehicleDatasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;

% 车辆数据存储在一个包含两列的表中，其中第一列包含图像文件路径，第二列包含车辆边界框
vehicleDataset(1:4,:)
vehicleDataset.imageFilename = fullfile(pwd,vehicleDataset.imageFilename);


% 将数据集分成训练集、验证集和测试集。
% 选择 60% 的数据用于训练，10% 用于验证，30% 用于测试
rng(0);
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * length(shuffledIndices) );

trainingIdx = 1:idx;
trainingDataTbl = vehicleDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = vehicleDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = vehicleDataset(shuffledIndices(testIdx),:);

% 使用 imageDatastore 和 boxLabelDatastore 创建数据存储，以便在训练和评估期间加载图像和标签数据。
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'vehicle'));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'vehicle'));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'vehicle'));

% 组合图像和边界框标签数据存储。
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

% 实例显示其中一个训练图像和边界框标签
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

% 创建 YOLO v2 目标检测网络
% YOLO v2 目标检测网络由两个子网络组成。一个特征提取网络，后跟一个检测网络
% 特征提取网络通常是一个预训练的 CNN， 这里使用 ResNet-50 进行特征提取
% 检测子网络是小型 CNN，它由几个卷积层和特定于 YOLO v2 的层组成

% 设定网络输入大小
inputSize = [224 224 3];

% 定义要检测的目标类的数量
numClasses = width(vehicleDataset)-1;

% 训练图像大于 224×224，并且大小不同，因此在训练前的预处理步骤中调整图像的大小
% 使用 estimateAnchorBoxes (Computer Vision Toolbox) 根据训练数据中目标的大小来估计锚框。
% 考虑到训练前会对图像大小进行调整，用来估计锚框的训练数据的大小也要调整。使用 transform 预处理训练数据，
% 然后定义锚框数量并估计锚框。使用支持函数 preprocessData 将训练数据的大小调整为网络的输入图像大小。
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)

% 使用 resnet50 加载预训练的 ResNet-50 模型
featureExtractionNetwork = resnet50;

% 选择 'activation_40_relu' 作为特征提取层，以将 'activation_40_relu' 后面的层替换为检测子网络。
featureLayer = 'activation_40_relu';

% 创建 YOLO v2 目标检测网络
lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);
% PS: 可以使用 analyzeNetwork 或者 Deep Learning Toolbox 中的 Deep Network Designer 来可视化网络。

% 数据增强
augmentedTrainingData = transform(trainingData,@augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
% 显示增强效果
figure
montage(augmentedData,'BorderSize',10)

% 预处理训练数据
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
preprocessedValidationData = transform(validationData,@(data)preprocessData(data,inputSize));
data = read(preprocessedTrainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

% 训练YOLO v2 目标检测模型
% 训练R-CNN参数设置
options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20,...
        'CheckpointPath',tempdir, ...
        'ValidationData',preprocessedValidationData);
% 训练R-CNN参数设置
if doTraining       
    % 训练模型
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
else
    % 加载预训练模型.
    pretrained = load('yolov2ResNet50VehicleExample_19b.mat');
    detector = pretrained.detector;
end

% 测试结果示例
I = imread('car.jpg');
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

% 使用测试集评估模型
% 使用平均精确率指标来评估性能
% 正确分类的能力（精确率）和模型找到所有相关对象的能力（召回率）
preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));
detectionResults = detect(detector, preprocessedTestData);
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))

% 相应的支持函数
function B = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.
B = cell(size(A));

I = A{1};
sz = size(I);
if numel(sz)==3 && sz(3) == 3
    I = jitterColorHSV(I,...
        'Contrast',0.2,...
        'Hue',0,...
        'Saturation',0.1,...
        'Brightness',0.2);
end

% Randomly flip and scale image.
tform = randomAffine2d('XReflection',true,'Scale',[1 1.1]);
rout = affineOutputView(sz,tform,'BoundsStyle','CenterOutput');
B{1} = imwarp(I,tform,'OutputView',rout);

% Apply same transform to boxes.
[B{2},indices] = bboxwarp(A{2},tform,rout,'OverlapThreshold',0.25);
B{3} = A{3}(indices);

% Return original data only when all boxes are removed by warping.
if isempty(indices)
    B = A;
end
end

function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to the targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
data{2} = bboxresize(data{2},scale);
end