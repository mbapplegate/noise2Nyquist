%%
%Calculate the NIQI for OCT conventionally processed
baseDir = '../results/oct/conventional/';
methods = {'median3','gaussian1','oofAvg3','bm3d25','bm4d25'};
scans = 0:35;
numIms = 96;
%%
%Start with the noisy images which will be a baseline
%Note only the central N frames are saved in this folder, so
%we can just go through all of them
NIQINoisy = zeros(length(scans),numIms);
fprintf('Working on the noisy Images...\n')
for scanNum = scans
    thisDir = fullfile(baseDir,'Images_none0',sprintf('%02d',scanNum));
    imList = dir(fullfile(thisDir,'*.png'));
    if scanNum == 10 %Scan 10 had no images
        NIQINoisy(scanNum+1,:) = 1;
    else
        for k = 1:numIms
            thisIm = imread(fullfile(thisDir,imList(k).name));
            thisNIQI = niqe(thisIm);
            NIQINoisy(scanNum+1,k) = thisNIQI;
        end
    end
end
%%Then do the other methods
%Pre-allocate for the new NIQI data
octNIQI = zeros(length(methods),length(scans),numIms);
%I will divide by this value to get relative NIQI
%This was easier for me than figuring out how to tile noisyNIQI
allNIQINoisy = zeros(size(octNIQI));
%Iterate through all methods
for i=1:length(methods)
    fprintf('Working on %s...\n',methods{i})
    %Copy NIQI noisy for this method
    allNIQINoisy(i,:,:) = NIQINoisy;
    for scanNum = scans
        %Skip scan 10 (no data)
        if scanNum == 10
            continue;
        end
        thisDir = fullfile(baseDir,sprintf('Images_%s',methods{i}),sprintf('%02d',scanNum));
        imList = dir(fullfile(thisDir,'*.png'));
        for k=1:numIms
            thisIm = imread(fullfile(thisDir,imList(k).name));
            thisNIQI = niqe(thisIm);
            octNIQI(i,scanNum+1,k) = thisNIQI;
        end
    end
end
%%
%Then analyze
NIQIRatio = octNIQI./allNIQINoisy;
NIQIRatio(:,11,:) = [];
save('../results/oct/NIQIRatio_Conventional.mat','NIQIRatio')
scanAvg = mean(NIQIRatio,3);
methodAvg = mean(scanAvg,2);
methodStd = std(scanAvg,1,2);
disp([methodAvg,methodStd])
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Repeat analysis for ML methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
learningMethods = {'noise2Nyq','noise2void','line2line','neighbor2neighbor'};
dataRuns = {'2022-07-15--13-45-14','2022-07-13--17-39-29','2022-07-19--03-17-51','neigh2neigh/2022-10-14-14-35'};
%Pre-allocate for the new NIQI data
MLNIQI = zeros(length(learningMethods),length(scans),numIms);
%I will divide by this value to get relative NIQI
originalNIQI = zeros(length(scans),numIms);
%This was easier for me than figuring out how to tile noisyNIQI
allMLNIQINoisy = zeros(size(MLNIQI));

%%
%First let's get a list of all the files I want to process
allImNames = cell(length(scans),numIms);
foldNum = zeros(1,length(scans));
foldIdx =1;
allStartFrames = zeros(1,length(scans));
for fold = 0:9
    dataDir = fullfile(sprintf('../results/oct/%s/%02d/testImages/last/',dataRuns{1},fold));
    [scansInFold,ims,startFrames] = getCenterNFramesOCT(dataDir,numIms);
    for i = 1:length(scansInFold)
        foldNum(scansInFold(i)+1) = fold;
        allStartFrames(scansInFold(i)+1) = startFrames(i);
        foldIdx = foldIdx+1;
        for j=1:numIms
            allImNames{scansInFold(i)+1,j} = ims{i,j};
        end
    end
end
%%
%Then get the NIQI for the original frames
originalDataDir = '/home/matthew/Documents/datasets/OCT_Denoise/Data';
%originalDataDir = 'D:\datasets\OCT Denoise\Data';
fprintf("Working on raw frames...\n");
for i = 1:length(scans)
    if scans(i) == 10
        originalNIQI(i,:) = 1;
        continue;
    end
    aviFile = fullfile(originalDataDir,sprintf('%02d.avi',scans(i)+1));
    vr = VideoReader(aviFile);
    for j=allStartFrames(i):allStartFrames(i)+numIms-1
        thisFrame = vr.read(j+1);
        thisNIQI = niqe(thisFrame(1:224,:,1)); %Decimation truncates image from 244 to 224
        originalNIQI(i,j-allStartFrames(i)+1) = thisNIQI;
    end
end

%%
%Then get the NIQI for the processed frames
%Iterate through all methods
for i=1:length(learningMethods)
    fprintf('Working on %s...\n',learningMethods{i})
    methodDir = fullfile(sprintf('../results/oct/%s/',dataRuns{i}));
    allMLNIQINoisy(i,:,:)=originalNIQI;
    for scanNum = scans
        %Skip scan 10 (no data)
        if scanNum == 10
            continue;
        end
        thisDir = fullfile(methodDir,sprintf('%02d',foldNum(scanNum+1)),'testImages','last');
        for k=1:numIms
            thisIm = imread(fullfile(thisDir,allImNames{scanNum+1,k}));
            thisNIQI = niqe(thisIm);
            MLNIQI(i,scanNum+1,k) = thisNIQI;
        end
    end
end

MLNIQIRatio = MLNIQI./allMLNIQINoisy;
MLNIQIRatio(:,11,:) = [];
save('../results/oct/NIQIRatio_MLMethods.mat','MLNIQIRatio')
scanAvgML = mean(MLNIQIRatio,3);
methodAvgML = mean(scanAvgML,2);
methodStdML = std(scanAvgML,1,2);
disp([methodAvgML,methodStdML])