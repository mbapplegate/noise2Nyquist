function [scansInFold,imNames,startFrames] = getRandomNFramesOCT(dataDir, numIms)
%getCenterNFramesOCT Return 2D stack of Image paths Scan x numFrames
%   Detailed explanation goes here
fList= dir(fullfile(dataDir,'*.png'));
scanNum = zeros(1,length(fList));
for i = 1:length(fList)
    fName = fList(i).name;
    scanNum(i) = str2double(fName(1:2));
end
scansInFold = unique(scanNum);

imNames=cell(length(scansInFold),numIms);
startFrames = zeros(1,length(scansInFold));
for i = 1:length(scansInFold)
    imsInScan = dir(fullfile(dataDir,sprintf('%02d*.png',scansInFold(i))));
    randIdx = randsample(length(imsInScan),numIms);
    imNames(i,:) = imsInScan(randIdx);
end
