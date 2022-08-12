%Code to make Shepp-Logan phantom and export it as a 3d volume so I can
%load it into Python and slice it up to test out the noise2noise on
%adjacent samples project
%
%M. Applegate
%2 June, 2022

eSL =  [  1  .6900  .920  .810      0       0       0      0      0      0
        -.8  .6624  .874  .780      0  -.0184       0      0      0      0
        -.15  .1100  .310  .220    .22       0       0    -18      0     10
        -.15  .1600  .410  .280   -.22       0       0     18      0     10
         .1  .2100  .250  .410      0     .35    -.15      0      0      0
         .1  .0460  .046  .050      0      .1     .25      0      0      0
         .1  .0460  .046  .050      0     -.1     .25      0      0      0
         .1  .0460  .023  .050   -.08   -.605       0      0      0      0
         .1  .0230  .023  .020      0   -.606       0      0      0      0
         .1  .0230  .046  .020    .06   -.605       0      0      0      0 ];

eYYW = [  1  .6900  .920  .900      0       0       0      0      0      0
        -.8  .6624  .874  .880      0       0       0      0      0      0
        -.15  .4100  .160  .210   -.22       0    -.25    108      0      0
        -.15  .3100  .110  .220    .22       0    -.25     72      0      0
         .2  .2100  .250  .500      0     .35    -.25      0      0      0
         .2  .0460  .046  .046      0      .1    -.25      0      0      0
         .1  .0460  .023  .020   -.08    -.65    -.25      0      0      0
         .1  .0460  .023  .020    .06    -.65    -.25     90      0      0
         .2  .0560  .040  .100    .06   -.105    .625     90      0      0
        -.15  .0560  .056  .100      0    .100    .625      0      0      0 ];     

phanSL= phantom3d(512,eSL);
phanYYW=phantom3d(512,eYYW);

%phan8Bit = round(phan*255);

save(fullfile('..','HRPhantomData','SheppLoganPhan.mat'),'phanSL');
save(fullfile('..','HRPhantomData','YuYeWangPhan.mat'),'phanYYW');

for i=1:64
    subplot(8,8,i)
    imshow(phanYYW(:,:,8*i)) 
end
