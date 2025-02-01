% Load preprocessed data
load('preprocessedData.mat', 'data', 'labels');

% Split data into training and validation sets
[trainInd, valInd] = dividerand(size(data, 4), 0.8, 0.2); % 80% training, 20% validation
trainData = data(:,:,:,trainInd);
trainLabels = labels(trainInd);
valData = data(:,:,:,valInd);
valLabels = labels(valInd);

% Define CNN layers
layers = [
    imageInputLayer([64 64 1]) % Input layer for grayscale images
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(numel(unique(labels))) % Output classes
    softmaxLayer
    classificationLayer
];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {valData, valLabels}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the model
trainedModel = trainNetwork(trainData, trainLabels, layers, options);

% Save the trained model
save('trainedDrowsinessModel.mat', 'trainedModel');
disp('Model training completed and saved.');
