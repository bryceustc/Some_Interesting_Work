clc;
clear all;
load('data_batch_1.mat');
load('data_batch_2.mat');
load('data_batch_3.mat');
load('data_batch_4.mat');
load('data_batch_5.mat');

for j=1:5
    load(['data_batch_' num2str(j) '.mat'])
    for i=1:size(data,1)
        p=data(i,:);
        label=labels(i);

        fig=zeros(32,32,3);
        fig(:,:,1)=reshape(p(1:1024),32,32)';
        fig(:,:,2)=reshape(p(1025:2048),32,32)';
        fig(:,:,3)=reshape(p(2049:end),32,32)';
        imwrite(fig/256,['batch_' num2str(j) '_label_' num2str(label) '_' num2str(i)  '.png'])
    end
end