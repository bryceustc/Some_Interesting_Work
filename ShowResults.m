function ShowResults(rcnn, testImage)

% 进行目标检测
[bboxes,score,label] = detect(rcnn,testImage,'MiniBatchSize',128);

% 显示检测结果
[score, idx] = max(score);
bbox = bboxes(idx, :);
annotation = sprintf('%s: (Confidence = %f)', label(idx), score);
outputImage = insertObjectAnnotation(testImage, 'rectangle', bbox, annotation);
figure
imshow(outputImage)
end