% Load the trained model
load('trainedDrowsinessModel.mat', 'trainedModel'); 

% Initialize webcam
cam = webcam;

% Check if webcam is successfully initialized
if isempty(cam)
    disp('Webcam not found. Please check your camera.');
    return;
end

% Create a face detector
faceDetector = vision.CascadeObjectDetector(); % Pre-trained face detector

% Set the image size (ensure it matches the input size of your model)
imageSize = [64, 64]; % Adjust to your model's input size

disp('Press Ctrl+C to stop live detection.');

% Initialize counters for eye and mouth states
eyeClosedConsecutiveFrames = 0; % Counter for eyes closed
mouthOpenConsecutiveFrames = 0; % Counter for mouth open

try
    while true
        % Capture a frame from the webcam
        frame = snapshot(cam);
        grayFrame = rgb2gray(frame); % Convert frame to grayscale

        % Check if the frame is valid
        if isempty(frame)
            disp('Failed to capture frame from webcam.');
            continue; % Skip this iteration if frame capture fails
        end
        
        % Attempt to detect a face
        bbox = step(faceDetector, grayFrame); % Get bounding box of the face
        
        % If no face is detected, display a message and continue
        if isempty(bbox)
            disp('No face detected. Trying again...');
            imshow(frame);
            title('No face detected.');
            drawnow;
            continue; % Skip the current loop iteration and try again
        end

        % Use the first detected face (if multiple faces are detected)
        faceRegion = bbox(1, :);

        % Define ROIs (Regions of Interest) dynamically based on the face bounding box
        faceX = faceRegion(1); % X-coordinate of the face
        faceY = faceRegion(2); % Y-coordinate of the face
        faceWidth = faceRegion(3); % Width of the face
        faceHeight = faceRegion(4); % Height of the face

        % Define left eye region (top-left part of the face)
        leftEyeRegion = imcrop(grayFrame, [faceX + 0.2*faceWidth, faceY + 0.25*faceHeight, 0.2*faceWidth, 0.2*faceHeight]);

        % Define right eye region (top-right part of the face)
        rightEyeRegion = imcrop(grayFrame, [faceX + 0.6*faceWidth, faceY + 0.25*faceHeight, 0.2*faceWidth, 0.2*faceHeight]);

        % Define mouth region (bottom-center part of the face)
        mouthRegion = imcrop(grayFrame, [faceX + 0.3*faceWidth, faceY + 0.7*faceHeight, 0.4*faceWidth, 0.2*faceHeight]);

        % Check if the regions are valid (non-empty)
        if ~isempty(leftEyeRegion) && ~isempty(rightEyeRegion) && ~isempty(mouthRegion)
            % Preprocess the ROIs for model input (resize them)
            leftEyeInput = imresize(leftEyeRegion, imageSize);
            rightEyeInput = imresize(rightEyeRegion, imageSize);
            mouthInput = imresize(mouthRegion, imageSize);

            % Normalize pixel values to [0, 1] for the model
            leftEyeInput = double(leftEyeInput) / 255;
            rightEyeInput = double(rightEyeInput) / 255;
            mouthInput = double(mouthInput) / 255;

            % Reshape to match the input format for the model
            leftEyeInput = reshape(leftEyeInput, [imageSize(1), imageSize(2), 1, 1]);
            rightEyeInput = reshape(rightEyeInput, [imageSize(1), imageSize(2), 1, 1]);
            mouthInput = reshape(mouthInput, [imageSize(1), imageSize(2), 1, 1]);

            % Classify the regions using the trained model
            leftEyePrediction = classify(trainedModel, leftEyeInput);
            rightEyePrediction = classify(trainedModel, rightEyeInput);
            mouthPrediction = classify(trainedModel, mouthInput);

            % Convert predictions to readable labels
            leftEyeLabel = char(leftEyePrediction);
            rightEyeLabel = char(rightEyePrediction);
            mouthLabel = char(mouthPrediction);

            % Check if both eyes are closed
            if strcmp(leftEyeLabel, 'LE_close') && strcmp(rightEyeLabel, 'RE_close')
                eyeClosedConsecutiveFrames = eyeClosedConsecutiveFrames + 1;
            else
                eyeClosedConsecutiveFrames = 0; % Reset if eyes are not both closed
            end

            % Check if mouth is open
            if strcmp(mouthLabel, 'M_open')
                mouthOpenConsecutiveFrames = mouthOpenConsecutiveFrames + 1;
            else
                mouthOpenConsecutiveFrames = 0; % Reset if mouth is not open
            end

            % Trigger alert if either condition is met
            if eyeClosedConsecutiveFrames >= 2
                disp('ALERT: The person is sleeping (both eyes closed for 2 consecutive frames).');
                eyeClosedConsecutiveFrames = 0; % Reset counter after alert
            end

            if mouthOpenConsecutiveFrames >= 2
                disp('ALERT: The person is sleeping (mouth open for 2 consecutive frames).');
                mouthOpenConsecutiveFrames = 0; % Reset counter after alert
            end

            % Annotate the detected regions
            annotatedFrame = insertObjectAnnotation(frame, 'rectangle', faceRegion, 'Face');
            annotatedFrame = insertObjectAnnotation(annotatedFrame, 'rectangle', [faceX + 0.2*faceWidth, faceY + 0.25*faceHeight, 0.2*faceWidth, 0.2*faceHeight], 'Left Eye');
            annotatedFrame = insertObjectAnnotation(annotatedFrame, 'rectangle', [faceX + 0.6*faceWidth, faceY + 0.25*faceHeight, 0.2*faceWidth, 0.2*faceHeight], 'Right Eye');
            annotatedFrame = insertObjectAnnotation(annotatedFrame, 'rectangle', [faceX + 0.3*faceWidth, faceY + 0.7*faceHeight, 0.4*faceWidth, 0.2*faceHeight], 'Mouth');

            % Show the annotated frame
            imshow(annotatedFrame);
            title(sprintf('Left Eye: %s | Right Eye: %s | Mouth: %s', ...
                leftEyeLabel, rightEyeLabel, mouthLabel));
        else
            disp('Unable to extract regions (face too small or cropped).');
        end
    % No face detected
    if isempty(bbox)
        disp('No face detected. Trying again...');
        imshow(frame);
        title('No face detected.');
        drawnow;
        continue; % Skip the current loop iteration and try again
    end
    drawnow;
end
catch ME
    % In case of error, release the webcam
    disp('Error occurred, releasing webcam...');
    rethrow(ME);
end

% Cleanup function to release the webcam
function releaseWebcam(cam)
    clear cam; % Clear webcam object
    disp('Webcam released.');
end
