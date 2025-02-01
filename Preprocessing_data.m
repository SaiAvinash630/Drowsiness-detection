% Load your dataset table
load('nestedImageDatabase.mat', 'datasetTable');

% Preprocess images: Resize and Normalize
imageSize = [64, 64]; % Resize all images to 64x64
data = [];
labels = [];

for i = 1:height(datasetTable)
    img = imread(datasetTable.ImagePath{i});
    img = imresize(img, imageSize); % Resize image
    img = double(img) / 255; % Normalize pixel values to [0, 1]
    data(:,:,:,i) = img; % Store image in 4D array
    labels{i} = datasetTable.Label{i}; % Store corresponding label
end

% Convert labels to categorical
labels = categorical(labels);

% Save preprocessed data
save('preprocessedData.mat', 'data', 'labels');
disp('Data preprocessing completed.');
