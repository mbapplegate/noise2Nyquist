function [scansInFold,imNames,startFrames] = getCenterNFramesOCT(dataDir, numIms)
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
    centerIm = round(length(imsInScan)/2);
    firstIm = floor(centerIm - numIms/2);
    lastIm = numIms + firstIm-1;
    splitsies = split(imsInScan(firstIm).name,'.');
    startFrames(i)=str2double(splitsies{1}(end-5:end-3));
    for j = firstIm:lastIm
        imNames{i,j-firstIm+1} = imsInScan(j).name;
    end
end
