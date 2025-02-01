% Define the root folder
rootFolder = 'D:\desktop\Drowsiness\Database'; % Replace with your dataset path

% Initialize empty cell arrays for storing data
imageData = {};
labels = {};

% Get the list of main folders
mainFolders = dir(rootFolder);
mainFolders = mainFolders([mainFolders.isdir]); % Only directories

% Loop through each main folder
for i = 1:length(mainFolders)
    mainFolderName = mainFolders(i).name;
    
    % Skip '.' and '..' folders
    if strcmp(mainFolderName, '.') || strcmp(mainFolderName, '..')
        continue;
    end
    
    % Full path of the main folder
    mainFolderPath = fullfile(rootFolder, mainFolderName);
    
    % Get the list of subfolders inside the main folder
    subFolders = dir(mainFolderPath);
    subFolders = subFolders([subFolders.isdir]); % Only directories
    
    % Loop through each subfolder
    for j = 1:length(subFolders)
        subFolderName = subFolders(j).name;
        
        % Skip '.' and '..' folders
        if strcmp(subFolderName, '.') || strcmp(subFolderName, '..')
            continue;
        end
        
        % Full path of the subfolder
        subFolderPath = fullfile(mainFolderPath, subFolderName);
        
        % Get the list of images in this subfolder
        images = dir(fullfile(subFolderPath, '*.bmp')); % Change extension if needed
        for k = 1:length(images)
            % Full path of the image
            imagePath = fullfile(images(k).folder, images(k).name);
            
            % Append image path and label to the database
            imageData{end+1, 1} = imagePath; % Store image path
            labels{end+1, 1} = strcat(mainFolderName, '_', subFolderName); % Store combined label
        end
    end
end

% Convert to a table for better handling
datasetTable = table(imageData, labels, 'VariableNames', {'ImagePath', 'Label'});

% Save the dataset to a .mat file
save('nestedImageDatabase.mat', 'datasetTable');

% Display the first few entries of the dataset
disp(head(datasetTable));
