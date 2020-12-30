%% Generation of Data Signals for Training of the CNN 
% MSc. Thesis - Rafael S. Salles - Federal University of Itajuba
% Supervisors: B. Isaias Lima Fuly and Paulo F. Ribeiro

clc
for i=1:200       
   set_param('IEEE5Bus/Capacitor Bank','CapacitivePower', ...
       num2str(abs(random('Normal',1.5e6,250e3))));
   set_param('IEEE5Bus/Lightning/Impulse Magnitude','Value', ...
       num2str(random('Normal',100e3,20e3)));
   set_param('IEEE5Bus/Voltage Sag','FaultResistance',num2str(abs(random('Normal',20,10))));
   set_param('IEEE5Bus/Interruption','FaultResistance',num2str(abs(random('Normal',3,0.5))));
   set_param('IEEE5Bus/LOAD 5','ActivePower',num2str(abs(random('Normal',100e6,30e6))));
   
   if(i<21)
   set_param('IEEE5Bus/CB 2','SwitchTimes','[5/60]');
   set_param('IEEE5Bus/Lightning/t1','Value','5/60');
   set_param('IEEE5Bus/Three-Phase Breaker','SwitchTimes','[5/60  10/60]');
   elseif(i>20 && i<41)
   set_param('IEEE5Bus/CB 2','SwitchTimes','[8/60]');
   set_param('IEEE5Bus/Lightning/t1','Value','8/60');
   set_param('IEEE5Bus/Three-Phase Breaker','SwitchTimes','[6/60  13/60]');
   elseif(i>40 && i<51)
   set_param('IEEE5Bus/Lightning/t1','Value','15/60');
   set_param('IEEE5Bus/CB 2','SwitchTimes','[13/60]');
   set_param('IEEE5Bus/Three-Phase Breaker','SwitchTimes','[11/60  18/60]');
   
   end
   
   sim('IEEE5Bus');
      
 sinal_pq(:,i)=ScopeData(:,4);
 sinal_30SNRdb(:,i)=addnoise(ScopeData(:,4),30);
 sinal_40SNRdb(:,i)=addnoise(ScopeData(:,4),40);
 sinal_60SNRdb(:,i)=addnoise(ScopeData(:,4),60);
         
end

 %% Add noise in SNR dB 
% MSc. Thesis - Rafael S. Salles - Federal University of Itajuba
% Supervisors: B. Isaias Lima Fuly and Paulo F. Ribeiro

function noisy_signal = addnoise( sig , SNRdb )

sig_power = mean(sig.^2);

SNR = 10^(SNRdb/10);

noisy_signal = sig + sqrt(sig_power/SNR)*randn(size(sig));

end

%% CWT with Filter Banks for 2-D Scalograms Extraction - Saving Images in Label Folders 
% MSc. Thesis - Rafael S. Salles - Federal University of Itajuba
% Supervisors: B. Isaias Lima Fuly and Paulo F. Ribeiro

clc
Fs = 200000; %sample frequency
fb = cwtfilterbank('SignalLength',56667,'SamplingFrequency',Fs, ...
    'VoicesPerOctave',48,'FrequencyLimits',[100 2000]); %cwtfilterbank
folder='C:\Users\rafae\Desktop\Dissertacao\imagens\interruption'; %folder of disturbance
for i=1:200
s=filtragem(sinal_pq(:,i)); % high-pass filter
cfs1= abs(fb.wt(s(10001:66667,1))); %cwt with filter bank
a=ind2rgb(im2uint8(rescale(cfs1)),jet(256)); %aplying color map
imdata(:,:,:,1)=imresize(a,[240 240]); %resize
outputFileName = fullfile(folder, [ 'interruption' num2str(i) '.jpg']); % file details
 imwrite(imdata(:,:,:,1), outputFileName); %sving file
%  
 s1=filtragem(sinal_30SNRdb(:,i));
 cfs2= abs(fb.wt(s1(10001:66667,1)));
a1=ind2rgb(im2uint8(rescale(cfs2)),jet(256));
imdata1(:,:,:,1)=imresize(a1,[240 240]); 
outputFileName1 = fullfile(folder, ['interruption30' num2str(i) '.jpg']);
 imwrite(imdata1(:,:,:,1), outputFileName1);
 
 s2=filtragem(sinal_40SNRdb(:,i));
 cfs3= abs(fb.wt(s2(10001:66667,1)));
a2=ind2rgb(im2uint8(rescale(cfs3)),jet(256));
imdata2(:,:,:,1)=imresize(a2,[240 240]); 
outputFileName2 = fullfile(folder, ['interruption40' num2str(i) '.jpg']);
 imwrite(imdata2(:,:,:,1), outputFileName2);
% %  
 s3=filtragem(sinal_60SNRdb(:,i));
 cfs4= abs(fb.wt(s3(10001:66667,1)));
a3=ind2rgb(im2uint8(rescale(cfs4)),jet(256));
imdata3(:,:,:,1)=imresize(a3,[240 240]); 
outputFileName3 = fullfile(folder, ['interruption60' num2str(i) '.jpg']);
 imwrite(imdata3(:,:,:,1), outputFileName3);
%  
end

%% Code for Design and Training - CNN from Scratch
% MSc. Thesis - Rafael S. Salles - Federal University of Itajuba
% Supervisors: B. Isaias Lima Fuly and Paulo F. Ribeiro

%Import training and validation data

imdsTrain = imageDatastore("C:\Users\rafae\Desktop\Dissertacao\imagens", ...
"IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain,0.8);

% Resize the images to match the network input layer.
augimdsTrain = augmentedImageDatastore([240 240 3],imdsTrain);
augimdsValidation = augmentedImageDatastore([240 240 3],imdsValidation);

%training options
opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.01,...
    "MaxEpochs",4,...
    "MiniBatchSize",24,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",10,...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);

%create an array of layers
layers = [
    imageInputLayer([240 240 3],"Name","imageinput")
    convolution2dLayer([3 8],32,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[2 2])
    convolution2dLayer([3 16],32,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[2 2])
    convolution2dLayer([3 32],32,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    maxPooling2dLayer([2 2],"Name","maxpool_3","Padding","same","Stride",[2 2])
    convolution2dLayer([3 64],32,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")
    maxPooling2dLayer([2 2],"Name","maxpool_4","Padding","same","Stride",[2 2])
    convolution2dLayer([3 128],32,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")
    fullyConnectedLayer(6,"Name","fc","BiasLearnRateFactor",10, ...
    "WeightLearnRateFactor",10)
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

%train cnn 
[net, traininfo] = trainNetwork(augimdsTrain,layers,opts);


%% Code for Design and Training - SqueezeNet
% MSc. Thesis - Rafael S. Salles - Federal University of Itajuba
% Supervisors: B. Isaias Lima Fuly and Paulo F. Ribeiro


%Import training and validation data
imdsTrain = imageDatastore("C:\Users\rafae\Desktop\Dissertacao\imagens", ...
    "IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain,0.8);

% Resize the images to match the network input layer.
augimdsTrain = augmentedImageDatastore([227 227 3],imdsTrain);
augimdsValidation = augmentedImageDatastore([227 227 3],imdsValidation);

%training options
opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.0001,...
    "MaxEpochs",4,...
    "MiniBatchSize",24,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",10,...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);

%create a layr graph with layers
lgraph = layerGraph();

%add the branches of the network to the layer graph
tempLayers = [
    imageInputLayer([227 227 3],"Name","data","Mean",trainingSetup.data.Mean)
    convolution2dLayer([3 3],64,"Name","conv1","Stride",[2 2], ...
    "Bias",trainingSetup.conv1.Bias,"Weights",trainingSetup.conv1.Weights)
    reluLayer("Name","relu_conv1")
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    convolution2dLayer([1 1],16,"Name","fire2-squeeze1x1", ...
    "Bias",trainingSetup.fire2_squeeze1x1.Bias, ...
    "Weights",trainingSetup.fire2_squeeze1x1.Weights)
    reluLayer("Name","fire2-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","fire2-expand3x3","Padding",[1 1 1 1], ...
    "Bias",trainingSetup.fire2_expand3x3.Bias,"Weights",trainingSetup.fire2_expand3x3.Weights)
    reluLayer("Name","fire2-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","fire2-expand1x1", ...
    "Bias",trainingSetup.fire2_expand1x1.Bias,"Weights",trainingSetup.fire2_expand1x1.Weights)
    reluLayer("Name","fire2-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire2-concat")
    convolution2dLayer([1 1],16,"Name","fire3-squeeze1x1", ...
    "Bias",trainingSetup.fire3_squeeze1x1.Bias, ...
    "Weights",trainingSetup.fire3_squeeze1x1.Weights)
    reluLayer("Name","fire3-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","fire3-expand3x3","Padding",[1 1 1 1], ...
    "Bias",trainingSetup.fire3_expand3x3.Bias,"Weights",trainingSetup.fire3_expand3x3.Weights)
    reluLayer("Name","fire3-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","fire3-expand1x1", ...
    "Bias",trainingSetup.fire3_expand1x1.Bias,"Weights",trainingSetup.fire3_expand1x1.Weights)
    reluLayer("Name","fire3-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire3-concat")
    maxPooling2dLayer([3 3],"Name","pool3","Padding",[0 1 0 1],"Stride",[2 2])
    convolution2dLayer([1 1],32,"Name","fire4-squeeze1x1", ...
    "Bias",trainingSetup.fire4_squeeze1x1.Bias, ...
    "Weights",trainingSetup.fire4_squeeze1x1.Weights)
    reluLayer("Name","fire4-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","fire4-expand1x1",... 
    "Bias",trainingSetup.fire4_expand1x1.Bias,"Weights",trainingSetup.fire4_expand1x1.Weights)
    reluLayer("Name","fire4-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","fire4-expand3x3","Padding",[1 1 1 1],... 
    "Bias",trainingSetup.fire4_expand3x3.Bias,"Weights",trainingSetup.fire4_expand3x3.Weights)
    reluLayer("Name","fire4-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire4-concat")
    convolution2dLayer([1 1],32,"Name","fire5-squeeze1x1", ...
    "Bias",trainingSetup.fire5_squeeze1x1.Bias, ...
    "Weights",trainingSetup.fire5_squeeze1x1.Weights)
    reluLayer("Name","fire5-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","fire5-expand1x1", ...
    "Bias",trainingSetup.fire5_expand1x1.Bias,"Weights",trainingSetup.fire5_expand1x1.Weights)
    reluLayer("Name","fire5-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","fire5-expand3x3","Padding",[1 1 1 1], ...
    "Bias",trainingSetup.fire5_expand3x3.Bias,"Weights",trainingSetup.fire5_expand3x3.Weights)
    reluLayer("Name","fire5-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire5-concat")
    maxPooling2dLayer([3 3],"Name","pool5","Padding",[0 1 0 1],"Stride",[2 2])
    convolution2dLayer([1 1],48,"Name","fire6-squeeze1x1", ...
    "Bias",trainingSetup.fire6_squeeze1x1.Bias, ...
    "Weights",trainingSetup.fire6_squeeze1x1.Weights)
    reluLayer("Name","fire6-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","fire6-expand3x3","Padding",[1 1 1 1], ... 
    "Bias",trainingSetup.fire6_expand3x3.Bias,"Weights",trainingSetup.fire6_expand3x3.Weights)
    reluLayer("Name","fire6-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","fire6-expand1x1", ... 
    "Bias",trainingSetup.fire6_expand1x1.Bias,"Weights",trainingSetup.fire6_expand1x1.Weights)
    reluLayer("Name","fire6-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire6-concat")
    convolution2dLayer([1 1],48,"Name","fire7-squeeze1x1", ... 
    "Bias",trainingSetup.fire7_squeeze1x1.Bias, ...
    "Weights",trainingSetup.fire7_squeeze1x1.Weights)
    reluLayer("Name","fire7-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","fire7-expand1x1", ...
    "Bias",trainingSetup.fire7_expand1x1.Bias,"Weights",trainingSetup.fire7_expand1x1.Weights)
    reluLayer("Name","fire7-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","fire7-expand3x3","Padding",[1 1 1 1],... 
    "Bias",trainingSetup.fire7_expand3x3.Bias,"Weights",trainingSetup.fire7_expand3x3.Weights)
    reluLayer("Name","fire7-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire7-concat")
    convolution2dLayer([1 1],64,"Name","fire8-squeeze1x1", ... 
    "Bias",trainingSetup.fire8_squeeze1x1.Bias, ...
    "Weights",trainingSetup.fire8_squeeze1x1.Weights)
    reluLayer("Name","fire8-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","fire8-expand3x3","Padding",[1 1 1 1],... 
    "Bias",trainingSetup.fire8_expand3x3.Bias,"Weights",trainingSetup.fire8_expand3x3.Weights)
    reluLayer("Name","fire8-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","fire8-expand1x1",... 
    "Bias",trainingSetup.fire8_expand1x1.Bias,"Weights",trainingSetup.fire8_expand1x1.Weights)
    reluLayer("Name","fire8-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire8-concat")
    convolution2dLayer([1 1],64,"Name","fire9-squeeze1x1",... 
    "Bias",trainingSetup.fire9_squeeze1x1.Bias, ...
    "Weights",trainingSetup.fire9_squeeze1x1.Weights)
    reluLayer("Name","fire9-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","fire9-expand1x1",... 
    "Bias",trainingSetup.fire9_expand1x1.Bias,"Weights",trainingSetup.fire9_expand1x1.Weights)
    reluLayer("Name","fire9-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","fire9-expand3x3","Padding",[1 1 1 1],... 
    "Bias",trainingSetup.fire9_expand3x3.Bias,"Weights",trainingSetup.fire9_expand3x3.Weights)
    reluLayer("Name","fire9-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire9-concat")
    dropoutLayer(0.5,"Name","drop9")
    convolution2dLayer([1 1],6,"Name","conv","Padding","same")
    reluLayer("Name","relu_conv10")
    globalAveragePooling2dLayer("Name","pool10")
    softmaxLayer("Name","prob")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

%connect the layers
lgraph = connectLayers(lgraph,"fire2-relu_squeeze1x1","fire2-expand3x3");
lgraph = connectLayers(lgraph,"fire2-relu_squeeze1x1","fire2-expand1x1");
lgraph = connectLayers(lgraph,"fire2-relu_expand3x3","fire2-concat/in2");
lgraph = connectLayers(lgraph,"fire2-relu_expand1x1","fire2-concat/in1");
lgraph = connectLayers(lgraph,"fire3-relu_squeeze1x1","fire3-expand3x3");
lgraph = connectLayers(lgraph,"fire3-relu_squeeze1x1","fire3-expand1x1");
lgraph = connectLayers(lgraph,"fire3-relu_expand3x3","fire3-concat/in2");
lgraph = connectLayers(lgraph,"fire3-relu_expand1x1","fire3-concat/in1");
lgraph = connectLayers(lgraph,"fire4-relu_squeeze1x1","fire4-expand1x1");
lgraph = connectLayers(lgraph,"fire4-relu_squeeze1x1","fire4-expand3x3");
lgraph = connectLayers(lgraph,"fire4-relu_expand1x1","fire4-concat/in1");
lgraph = connectLayers(lgraph,"fire4-relu_expand3x3","fire4-concat/in2");
lgraph = connectLayers(lgraph,"fire5-relu_squeeze1x1","fire5-expand1x1");
lgraph = connectLayers(lgraph,"fire5-relu_squeeze1x1","fire5-expand3x3");
lgraph = connectLayers(lgraph,"fire5-relu_expand1x1","fire5-concat/in1");
lgraph = connectLayers(lgraph,"fire5-relu_expand3x3","fire5-concat/in2");
lgraph = connectLayers(lgraph,"fire6-relu_squeeze1x1","fire6-expand3x3");
lgraph = connectLayers(lgraph,"fire6-relu_squeeze1x1","fire6-expand1x1");
lgraph = connectLayers(lgraph,"fire6-relu_expand1x1","fire6-concat/in1");
lgraph = connectLayers(lgraph,"fire6-relu_expand3x3","fire6-concat/in2");
lgraph = connectLayers(lgraph,"fire7-relu_squeeze1x1","fire7-expand1x1");
lgraph = connectLayers(lgraph,"fire7-relu_squeeze1x1","fire7-expand3x3");
lgraph = connectLayers(lgraph,"fire7-relu_expand1x1","fire7-concat/in1");
lgraph = connectLayers(lgraph,"fire7-relu_expand3x3","fire7-concat/in2");
lgraph = connectLayers(lgraph,"fire8-relu_squeeze1x1","fire8-expand3x3");
lgraph = connectLayers(lgraph,"fire8-relu_squeeze1x1","fire8-expand1x1");
lgraph = connectLayers(lgraph,"fire8-relu_expand3x3","fire8-concat/in2");
lgraph = connectLayers(lgraph,"fire8-relu_expand1x1","fire8-concat/in1");
lgraph = connectLayers(lgraph,"fire9-relu_squeeze1x1","fire9-expand1x1");
lgraph = connectLayers(lgraph,"fire9-relu_squeeze1x1","fire9-expand3x3");
lgraph = connectLayers(lgraph,"fire9-relu_expand1x1","fire9-concat/in1");
lgraph = connectLayers(lgraph,"fire9-relu_expand3x3","fire9-concat/in2");

%train cnn
[net, traininfo] = trainNetwork(augimdsTrain,lgraph,opts);

%% Code for Design and Training - GoogleNet
% MSc. Thesis - Rafael S. Salles - Federal University of Itajuba
% Supervisors: B. Isaias Lima Fuly and Paulo F. Ribeiro


%Import training and validation data
imdsTrain = imageDatastore("C:\Users\rafae\Desktop\Dissertacao\imagens", ...
"IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain,0.8);

% Resize the images to match the network input layer.
augimdsTrain = augmentedImageDatastore([224 224 3],imdsTrain);
augimdsValidation = augmentedImageDatastore([224 224 3],imdsValidation);

%training options
opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.0001,...
    "MaxEpochs",4,...
    "MiniBatchSize",24,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",10,...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation);

%create a layr graph with layers
lgraph = layerGraph();

%add the branches of the network to the layer graph
tempLayers = [
    imageInputLayer([224 224 3],"Name","data","Mean",trainingSetup.data.Mean)
    convolution2dLayer([7 7],64,"Name","conv1-7x7_s2",...
    "BiasLearnRateFactor",2,"Padding",[3 3 3 3],"Stride",[2 2],...
    "Bias",trainingSetup.conv1_7x7_s2.Bias,"Weights",trainingSetup.conv1_7x7_s2.Weights)
    reluLayer("Name","conv1-relu_7x7")
    maxPooling2dLayer([3 3],"Name","pool1-3x3_s2","Padding",[0 1 0 1],"Stride",[2 2])
    crossChannelNormalizationLayer(5,"Name","pool1-norm1","K",1)
    convolution2dLayer([1 1],64,"Name","conv2-3x3_reduce",...
    "BiasLearnRateFactor",2,"Bias",trainingSetup.conv2_3x3_reduce.Bias,...
    "Weights",trainingSetup.conv2_3x3_reduce.Weights)
    reluLayer("Name","conv2-relu_3x3_reduce")
    convolution2dLayer([3 3],192,"Name","conv2-3x3",...
    "BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.conv2_3x3.Bias,...
    "Weights",trainingSetup.conv2_3x3.Weights)
    reluLayer("Name","conv2-relu_3x3")
    crossChannelNormalizationLayer(5,"Name","conv2-norm2","K",1)
    maxPooling2dLayer([3 3],"Name","pool2-3x3_s2","Padding",[0 1 0 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],96,"Name","inception_3a-3x3_reduce",...
    "BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_3a_3x3_reduce.Bias,...
    "Weights",trainingSetup.inception_3a_3x3_reduce.Weights)
    reluLayer("Name","inception_3a-relu_3x3_reduce")
    convolution2dLayer([3 3],128,"Name","inception_3a-3x3",...
    "BiasLearnRateFactor",2,"Padding",[1 1 1 1],...
    "Bias",trainingSetup.inception_3a_3x3.Bias,...
    "Weights",trainingSetup.inception_3a_3x3.Weights)
    reluLayer("Name","inception_3a-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","inception_3a-1x1","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_3a_1x1.Bias,...
    "Weights",trainingSetup.inception_3a_1x1.Weights)
    reluLayer("Name","inception_3a-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","inception_3a-5x5_reduce",...
    "BiasLearnRateFactor",2,"Bias",trainingSetup.inception_3a_5x5_reduce.Bias,...
    "Weights",trainingSetup.inception_3a_5x5_reduce.Weights)
    reluLayer("Name","inception_3a-relu_5x5_reduce")
    convolution2dLayer([5 5],32,"Name","inception_3a-5x5","BiasLearnRateFactor",2,...
    "Padding",[2 2 2 2],"Bias",trainingSetup.inception_3a_5x5.Bias,...
    "Weights",trainingSetup.inception_3a_5x5.Weights)
    reluLayer("Name","inception_3a-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_3a-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],32,"Name","inception_3a-pool_proj","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_3a_pool_proj.Bias,...
    "Weights",trainingSetup.inception_3a_pool_proj.Weights)
    reluLayer("Name","inception_3a-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_3a-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_3b-5x5_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_3b_5x5_reduce.Bias,...
    "Weights",trainingSetup.inception_3b_5x5_reduce.Weights)
    reluLayer("Name","inception_3b-relu_5x5_reduce")
    convolution2dLayer([5 5],96,"Name","inception_3b-5x5",...
    "BiasLearnRateFactor",2,"Padding",[2 2 2 2],...
    "Bias",trainingSetup.inception_3b_5x5.Bias,...
    "Weights",trainingSetup.inception_3b_5x5.Weights)
    reluLayer("Name","inception_3b-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_3b-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],64,"Name","inception_3b-pool_proj",...
    "BiasLearnRateFactor",2,"Bias",trainingSetup.inception_3b_pool_proj.Bias,...
    "Weights",trainingSetup.inception_3b_pool_proj.Weights)
    reluLayer("Name","inception_3b-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","inception_3b-1x1","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_3b_1x1.Bias,...
    "Weights",trainingSetup.inception_3b_1x1.Weights)
    reluLayer("Name","inception_3b-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","inception_3b-3x3_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_3b_3x3_reduce.Bias,...
    "Weights",trainingSetup.inception_3b_3x3_reduce.Weights)
    reluLayer("Name","inception_3b-relu_3x3_reduce")
    convolution2dLayer([3 3],192,"Name","inception_3b-3x3","BiasLearnRateFactor",2,...
    "Padding",[1 1 1 1],"Bias",trainingSetup.inception_3b_3x3.Bias,...
    "Weights",trainingSetup.inception_3b_3x3.Weights)
    reluLayer("Name","inception_3b-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","inception_3b-output")
    maxPooling2dLayer([3 3],"Name","pool3-3x3_s2","Padding",[0 1 0 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_4a-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],64,"Name","inception_4a-pool_proj","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4a_pool_proj.Bias,...
    "Weights",trainingSetup.inception_4a_pool_proj.Weights)
    reluLayer("Name","inception_4a-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","inception_4a-1x1","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4a_1x1.Bias,"Weights",trainingSetup.inception_4a_1x1.Weights)
    reluLayer("Name","inception_4a-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","inception_4a-5x5_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4a_5x5_reduce.Bias,...
    "Weights",trainingSetup.inception_4a_5x5_reduce.Weights)
    reluLayer("Name","inception_4a-relu_5x5_reduce")
    convolution2dLayer([5 5],48,"Name","inception_4a-5x5","BiasLearnRateFactor",2,...
    "Padding",[2 2 2 2],"Bias",trainingSetup.inception_4a_5x5.Bias,...
    "Weights",trainingSetup.inception_4a_5x5.Weights)
    reluLayer("Name","inception_4a-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],96,"Name","inception_4a-3x3_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4a_3x3_reduce.Bias,...
    "Weights",trainingSetup.inception_4a_3x3_reduce.Weights)
    reluLayer("Name","inception_4a-relu_3x3_reduce")
    convolution2dLayer([3 3],208,"Name","inception_4a-3x3",...
    "BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.inception_4a_3x3.Bias,...
    "Weights",trainingSetup.inception_4a_3x3.Weights)
    reluLayer("Name","inception_4a-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_4a-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],112,"Name","inception_4b-3x3_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4b_3x3_reduce.Bias,...
    "Weights",trainingSetup.inception_4b_3x3_reduce.Weights)
    reluLayer("Name","inception_4b-relu_3x3_reduce")
    convolution2dLayer([3 3],224,"Name","inception_4b-3x3","BiasLearnRateFactor",2,...
    "Padding",[1 1 1 1],"Bias",trainingSetup.inception_4b_3x3.Bias,...
    "Weights",trainingSetup.inception_4b_3x3.Weights)
    reluLayer("Name","inception_4b-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_4b-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],64,"Name","inception_4b-pool_proj","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4b_pool_proj.Bias,...
    "Weights",trainingSetup.inception_4b_pool_proj.Weights)
    reluLayer("Name","inception_4b-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],24,"Name","inception_4b-5x5_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4b_5x5_reduce.Bias,...
    "Weights",trainingSetup.inception_4b_5x5_reduce.Weights)
    reluLayer("Name","inception_4b-relu_5x5_reduce")
    convolution2dLayer([5 5],64,"Name","inception_4b-5x5","BiasLearnRateFactor",2,...
    "Padding",[2 2 2 2],"Bias",trainingSetup.inception_4b_5x5.Bias,...
    "Weights",trainingSetup.inception_4b_5x5.Weights)
    reluLayer("Name","inception_4b-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],160,"Name","inception_4b-1x1","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4b_1x1.Bias,...
    "Weights",trainingSetup.inception_4b_1x1.Weights)
    reluLayer("Name","inception_4b-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_4b-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","inception_4c-1x1","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4c_1x1.Bias,...
    "Weights",trainingSetup.inception_4c_1x1.Weights)
    reluLayer("Name","inception_4c-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","inception_4c-3x3_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4c_3x3_reduce.Bias,...
    "Weights",trainingSetup.inception_4c_3x3_reduce.Weights)
    reluLayer("Name","inception_4c-relu_3x3_reduce")
    convolution2dLayer([3 3],256,"Name","inception_4c-3x3","BiasLearnRateFactor",2,...
    "Padding",[1 1 1 1],"Bias",trainingSetup.inception_4c_3x3.Bias,...
    "Weights",trainingSetup.inception_4c_3x3.Weights)
    reluLayer("Name","inception_4c-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],24,"Name","inception_4c-5x5_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4c_5x5_reduce.Bias,...
    "Weights",trainingSetup.inception_4c_5x5_reduce.Weights)
    reluLayer("Name","inception_4c-relu_5x5_reduce")
    convolution2dLayer([5 5],64,"Name","inception_4c-5x5","BiasLearnRateFactor",2,...
    "Padding",[2 2 2 2],"Bias",trainingSetup.inception_4c_5x5.Bias,...
    "Weights",trainingSetup.inception_4c_5x5.Weights)
    reluLayer("Name","inception_4c-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_4c-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],64,"Name","inception_4c-pool_proj","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4c_pool_proj.Bias,...
    "Weights",trainingSetup.inception_4c_pool_proj.Weights)
    reluLayer("Name","inception_4c-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_4c-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],112,"Name","inception_4d-1x1","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4d_1x1.Bias,...
    "Weights",trainingSetup.inception_4d_1x1.Weights)
    reluLayer("Name","inception_4d-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],144,"Name","inception_4d-3x3_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4d_3x3_reduce.Bias,...
    "Weights",trainingSetup.inception_4d_3x3_reduce.Weights)
    reluLayer("Name","inception_4d-relu_3x3_reduce")
    convolution2dLayer([3 3],288,"Name","inception_4d-3x3","BiasLearnRateFactor",2,...
    "Padding",[1 1 1 1],"Bias",trainingSetup.inception_4d_3x3.Bias,...
    "Weights",trainingSetup.inception_4d_3x3.Weights)
    reluLayer("Name","inception_4d-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_4d-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],64,"Name","inception_4d-pool_proj","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4d_pool_proj.Bias,...
    "Weights",trainingSetup.inception_4d_pool_proj.Weights)
    reluLayer("Name","inception_4d-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_4d-5x5_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4d_5x5_reduce.Bias,...
    "Weights",trainingSetup.inception_4d_5x5_reduce.Weights)
    reluLayer("Name","inception_4d-relu_5x5_reduce")
    convolution2dLayer([5 5],64,"Name","inception_4d-5x5","BiasLearnRateFactor",2,...
    "Padding",[2 2 2 2],"Bias",trainingSetup.inception_4d_5x5.Bias,...
    "Weights",trainingSetup.inception_4d_5x5.Weights)
    reluLayer("Name","inception_4d-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_4d-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_4e-5x5_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4e_5x5_reduce.Bias,...
    "Weights",trainingSetup.inception_4e_5x5_reduce.Weights)
    reluLayer("Name","inception_4e-relu_5x5_reduce")
    convolution2dLayer([5 5],128,"Name","inception_4e-5x5","BiasLearnRateFactor",2,...
    "Padding",[2 2 2 2],"Bias",trainingSetup.inception_4e_5x5.Bias,...
    "Weights",trainingSetup.inception_4e_5x5.Weights)
    reluLayer("Name","inception_4e-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],160,"Name","inception_4e-3x3_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4e_3x3_reduce.Bias,...
    "Weights",trainingSetup.inception_4e_3x3_reduce.Weights)
    reluLayer("Name","inception_4e-relu_3x3_reduce")
    convolution2dLayer([3 3],320,"Name","inception_4e-3x3","BiasLearnRateFactor",2,...
    "Padding",[1 1 1 1],"Bias",trainingSetup.inception_4e_3x3.Bias,...
    "Weights",trainingSetup.inception_4e_3x3.Weights)
    reluLayer("Name","inception_4e-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","inception_4e-1x1","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4e_1x1.Bias,...
    "Weights",trainingSetup.inception_4e_1x1.Weights)
    reluLayer("Name","inception_4e-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_4e-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],128,"Name","inception_4e-pool_proj","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_4e_pool_proj.Bias,...
    "Weights",trainingSetup.inception_4e_pool_proj.Weights)
    reluLayer("Name","inception_4e-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","inception_4e-output")
    maxPooling2dLayer([3 3],"Name","pool4-3x3_s2","Padding",[0 1 0 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],160,"Name","inception_5a-3x3_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_5a_3x3_reduce.Bias,...
    "Weights",trainingSetup.inception_5a_3x3_reduce.Weights)
    reluLayer("Name","inception_5a-relu_3x3_reduce")
    convolution2dLayer([3 3],320,"Name","inception_5a-3x3","BiasLearnRateFactor",2,...
    "Padding",[1 1 1 1],"Bias",trainingSetup.inception_5a_3x3.Bias,...
    "Weights",trainingSetup.inception_5a_3x3.Weights)
    reluLayer("Name","inception_5a-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_5a-5x5_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_5a_5x5_reduce.Bias,...
    "Weights",trainingSetup.inception_5a_5x5_reduce.Weights)
    reluLayer("Name","inception_5a-relu_5x5_reduce")
    convolution2dLayer([5 5],128,"Name","inception_5a-5x5","BiasLearnRateFactor",2,...
    "Padding",[2 2 2 2],"Bias",trainingSetup.inception_5a_5x5.Bias,...
    "Weights",trainingSetup.inception_5a_5x5.Weights)
    reluLayer("Name","inception_5a-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_5a-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],128,"Name","inception_5a-pool_proj","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_5a_pool_proj.Bias,...
    "Weights",trainingSetup.inception_5a_pool_proj.Weights)
    reluLayer("Name","inception_5a-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","inception_5a-1x1","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_5a_1x1.Bias,...
    "Weights",trainingSetup.inception_5a_1x1.Weights)
    reluLayer("Name","inception_5a-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_5a-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","inception_5b-3x3_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_5b_3x3_reduce.Bias,...
    "Weights",trainingSetup.inception_5b_3x3_reduce.Weights)
    reluLayer("Name","inception_5b-relu_3x3_reduce")
    convolution2dLayer([3 3],384,"Name","inception_5b-3x3","BiasLearnRateFactor",2,...
    "Padding",[1 1 1 1],"Bias",trainingSetup.inception_5b_3x3.Bias,...
    "Weights",trainingSetup.inception_5b_3x3.Weights)
    reluLayer("Name","inception_5b-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","inception_5b-1x1","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_5b_1x1.Bias,...
    "Weights",trainingSetup.inception_5b_1x1.Weights)
    reluLayer("Name","inception_5b-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_5b-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],128,"Name","inception_5b-pool_proj","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_5b_pool_proj.Bias,...
    "Weights",trainingSetup.inception_5b_pool_proj.Weights)
    reluLayer("Name","inception_5b-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],48,"Name","inception_5b-5x5_reduce","BiasLearnRateFactor",2,...
    "Bias",trainingSetup.inception_5b_5x5_reduce.Bias,...
    "Weights",trainingSetup.inception_5b_5x5_reduce.Weights)
    reluLayer("Name","inception_5b-relu_5x5_reduce")
    convolution2dLayer([5 5],128,"Name","inception_5b-5x5","BiasLearnRateFactor",2,...
    "Padding",[2 2 2 2],"Bias",trainingSetup.inception_5b_5x5.Bias,...
    "Weights",trainingSetup.inception_5b_5x5.Weights)
    reluLayer("Name","inception_5b-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","inception_5b-output")
    globalAveragePooling2dLayer("Name","pool5-7x7_s1")
    dropoutLayer(0.4,"Name","pool5-drop_7x7_s1")
    fullyConnectedLayer(6,"Name","fc","BiasLearnRateFactor",10,"WeightLearnRateFactor",10)
    softmaxLayer("Name","prob")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
%connect braches
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-3x3_reduce");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-1x1");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-5x5_reduce");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-pool");
lgraph = connectLayers(lgraph,"inception_3a-relu_3x3","inception_3a-output/in2");
lgraph = connectLayers(lgraph,"inception_3a-relu_1x1","inception_3a-output/in1");
lgraph = connectLayers(lgraph,"inception_3a-relu_5x5","inception_3a-output/in3");
lgraph = connectLayers(lgraph,"inception_3a-relu_pool_proj","inception_3a-output/in4");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-pool");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-1x1");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_3b-relu_1x1","inception_3b-output/in1");
lgraph = connectLayers(lgraph,"inception_3b-relu_pool_proj","inception_3b-output/in4");
lgraph = connectLayers(lgraph,"inception_3b-relu_3x3","inception_3b-output/in2");
lgraph = connectLayers(lgraph,"inception_3b-relu_5x5","inception_3b-output/in3");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-pool");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-1x1");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-5x5_reduce");
lgraph = connectLayers(lgraph,"pool3-3x3_s2","inception_4a-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4a-relu_1x1","inception_4a-output/in1");
lgraph = connectLayers(lgraph,"inception_4a-relu_pool_proj","inception_4a-output/in4");
lgraph = connectLayers(lgraph,"inception_4a-relu_5x5","inception_4a-output/in3");
lgraph = connectLayers(lgraph,"inception_4a-relu_3x3","inception_4a-output/in2");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-pool");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_4a-output","inception_4b-1x1");
lgraph = connectLayers(lgraph,"inception_4b-relu_3x3","inception_4b-output/in2");
lgraph = connectLayers(lgraph,"inception_4b-relu_5x5","inception_4b-output/in3");
lgraph = connectLayers(lgraph,"inception_4b-relu_pool_proj","inception_4b-output/in4");
lgraph = connectLayers(lgraph,"inception_4b-relu_1x1","inception_4b-output/in1");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-1x1");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_4b-output","inception_4c-pool");
lgraph = connectLayers(lgraph,"inception_4c-relu_1x1","inception_4c-output/in1");
lgraph = connectLayers(lgraph,"inception_4c-relu_3x3","inception_4c-output/in2");
lgraph = connectLayers(lgraph,"inception_4c-relu_pool_proj","inception_4c-output/in4");
lgraph = connectLayers(lgraph,"inception_4c-relu_5x5","inception_4c-output/in3");
lgraph = connectLayers(lgraph,"inception_4c-output","inception_4d-1x1");
lgraph = connectLayers(lgraph,"inception_4c-output","inception_4d-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4c-output","inception_4d-pool");
lgraph = connectLayers(lgraph,"inception_4c-output","inception_4d-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_4d-relu_1x1","inception_4d-output/in1");
lgraph = connectLayers(lgraph,"inception_4d-relu_pool_proj","inception_4d-output/in4");
lgraph = connectLayers(lgraph,"inception_4d-relu_3x3","inception_4d-output/in2");
lgraph = connectLayers(lgraph,"inception_4d-relu_5x5","inception_4d-output/in3");
lgraph = connectLayers(lgraph,"inception_4d-output","inception_4e-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_4d-output","inception_4e-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_4d-output","inception_4e-1x1");
lgraph = connectLayers(lgraph,"inception_4d-output","inception_4e-pool");
lgraph = connectLayers(lgraph,"inception_4e-relu_1x1","inception_4e-output/in1");
lgraph = connectLayers(lgraph,"inception_4e-relu_pool_proj","inception_4e-output/in4");
lgraph = connectLayers(lgraph,"inception_4e-relu_5x5","inception_4e-output/in3");
lgraph = connectLayers(lgraph,"inception_4e-relu_3x3","inception_4e-output/in2");
lgraph = connectLayers(lgraph,"pool4-3x3_s2","inception_5a-3x3_reduce");
lgraph = connectLayers(lgraph,"pool4-3x3_s2","inception_5a-5x5_reduce");
lgraph = connectLayers(lgraph,"pool4-3x3_s2","inception_5a-pool");
lgraph = connectLayers(lgraph,"pool4-3x3_s2","inception_5a-1x1");
lgraph = connectLayers(lgraph,"inception_5a-relu_3x3","inception_5a-output/in2");
lgraph = connectLayers(lgraph,"inception_5a-relu_pool_proj","inception_5a-output/in4");
lgraph = connectLayers(lgraph,"inception_5a-relu_5x5","inception_5a-output/in3");
lgraph = connectLayers(lgraph,"inception_5a-relu_1x1","inception_5a-output/in1");
lgraph = connectLayers(lgraph,"inception_5a-output","inception_5b-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_5a-output","inception_5b-1x1");
lgraph = connectLayers(lgraph,"inception_5a-output","inception_5b-pool");
lgraph = connectLayers(lgraph,"inception_5a-output","inception_5b-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_5b-relu_1x1","inception_5b-output/in1");
lgraph = connectLayers(lgraph,"inception_5b-relu_pool_proj","inception_5b-output/in4");
lgraph = connectLayers(lgraph,"inception_5b-relu_3x3","inception_5b-output/in2");
lgraph = connectLayers(lgraph,"inception_5b-relu_5x5","inception_5b-output/in3");

%train cnn
[net, traininfo] = trainNetwork(augimdsTrain,lgraph,opts);

%% Code for Design and Training - ResNet-50
% MSc. Thesis - Rafael S. Salles - Federal University of Itajuba
% Supervisors: B. Isaias Lima Fuly and Paulo F. Ribeiro


%Import training and validation data
imdsTrain = imageDatastore("C:\Users\rafae\Desktop\Dissertacao\imagens",...
"IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain,0.8);

% Resize the images to match the network input layer.
augimdsTrain = augmentedImageDatastore([224 224 3],imdsTrain);
augimdsValidation = augmentedImageDatastore([224 224 3],imdsValidation);

%training options
opts = trainingOptions("sgdm",...
"ExecutionEnvironment","auto",...
"InitialLearnRate",0.0001,...
"MaxEpochs",4,...
"MiniBatchSize",24,...
"Shuffle","every-epoch",...
"ValidationFrequency",10,...
"Plots","training-progress",...
"ValidationData",augimdsValidation);

%create a layr graph with layers
lgraph = layerGraph();

%add the branches of the network to the layer graph
tempLayers = [
imageInputLayer([224 224 3],"Name","input_1","Mean",trainingSetup.input_1.Mean)
convolution2dLayer([7 7],64,"Name","conv1","Padding",[3 3 3 3],"Stride",[2 2],...
"Bias",trainingSetup.conv1.Bias,"Weights",trainingSetup.conv1.Weights)
batchNormalizationLayer("Name","bn_conv1","Epsilon",0.001, ...
"Offset",trainingSetup.bn_conv1.Offset,"Scale",trainingSetup.bn_conv1.Scale,...
"TrainedMean",trainingSetup.bn_conv1.TrainedMean,...
"TrainedVariance",trainingSetup.bn_conv1.TrainedVariance)
reluLayer("Name","activation_1_relu")
maxPooling2dLayer([3 3],"Name","max_pooling2d_1","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],64,"Name","res2a_branch2a","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res2a_branch2a.Bias,"Weights",trainingSetup.res2a_branch2a.Weights)
batchNormalizationLayer("Name","bn2a_branch2a","Epsilon",0.001,...
"Offset",trainingSetup.bn2a_branch2a.Offset, ...
"Scale",trainingSetup.bn2a_branch2a.Scale,...
"TrainedMean",trainingSetup.bn2a_branch2a.TrainedMean, ...
"TrainedVariance",trainingSetup.bn2a_branch2a.TrainedVariance)
reluLayer("Name","activation_2_relu")
convolution2dLayer([3 3],64,"Name","res2a_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res2a_branch2b.Bias,"Weights",trainingSetup.res2a_branch2b.Weights)
batchNormalizationLayer("Name","bn2a_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn2a_branch2b.Offset,"Scale",trainingSetup.bn2a_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn2a_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn2a_branch2b.TrainedVariance)
reluLayer("Name","activation_3_relu")
convolution2dLayer([1 1],256,"Name","res2a_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res2a_branch2c.Bias,"Weights",trainingSetup.res2a_branch2c.Weights)
batchNormalizationLayer("Name","bn2a_branch2c","Epsilon",0.001, ...
"Offset",trainingSetup.bn2a_branch2c.Offset,"Scale",trainingSetup.bn2a_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn2a_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn2a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],256,"Name","res2a_branch1","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res2a_branch1.Bias,"Weights",trainingSetup.res2a_branch1.Weights)
batchNormalizationLayer("Name","bn2a_branch1","Epsilon",0.001, ...
"Offset",trainingSetup.bn2a_branch1.Offset,...
"Scale",trainingSetup.bn2a_branch1.Scale, ...
"TrainedMean",trainingSetup.bn2a_branch1.TrainedMean,...
"TrainedVariance",trainingSetup.bn2a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_1")
reluLayer("Name","activation_4_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],64,"Name","res2b_branch2a","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res2b_branch2a.Bias,"Weights",trainingSetup.res2b_branch2a.Weights)
batchNormalizationLayer("Name","bn2b_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn2b_branch2a.Offset,"Scale",trainingSetup.bn2b_branch2a.Scale, ...
"TrainedMean",trainingSetup.bn2b_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn2b_branch2a.TrainedVariance)
reluLayer("Name","activation_5_relu")
convolution2dLayer([3 3],64,"Name","res2b_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res2b_branch2b.Bias,"Weights",trainingSetup.res2b_branch2b.Weights)
batchNormalizationLayer("Name","bn2b_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn2b_branch2b.Offset,...
"Scale",trainingSetup.bn2b_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn2b_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn2b_branch2b.TrainedVariance)
reluLayer("Name","activation_6_relu")
convolution2dLayer([1 1],256,"Name","res2b_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res2b_branch2c.Bias,"Weights",trainingSetup.res2b_branch2c.Weights)
batchNormalizationLayer("Name","bn2b_branch2c","Epsilon",0.001, ...
"Offset",trainingSetup.bn2b_branch2c.Offset,...
"Scale",trainingSetup.bn2b_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn2b_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn2b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_2")
reluLayer("Name","activation_7_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],64,"Name","res2c_branch2a","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res2c_branch2a.Bias,"Weights",trainingSetup.res2c_branch2a.Weights)
batchNormalizationLayer("Name","bn2c_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn2c_branch2a.Offset,...
"Scale",trainingSetup.bn2c_branch2a.Scale, ...
"TrainedMean",trainingSetup.bn2c_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn2c_branch2a.TrainedVariance)
reluLayer("Name","activation_8_relu")
convolution2dLayer([3 3],64,"Name","res2c_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res2c_branch2b.Bias,"Weights",trainingSetup.res2c_branch2b.Weights)
batchNormalizationLayer("Name","bn2c_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn2c_branch2b.Offset,...
"Scale",trainingSetup.bn2c_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn2c_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn2c_branch2b.TrainedVariance)
reluLayer("Name","activation_9_relu")
convolution2dLayer([1 1],256,"Name","res2c_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res2c_branch2c.Bias,"Weights",trainingSetup.res2c_branch2c.Weights)
batchNormalizationLayer("Name","bn2c_branch2c","Epsilon",0.001, ...
"Offset",trainingSetup.bn2c_branch2c.Offset,...
"Scale",trainingSetup.bn2c_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn2c_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn2c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_3")
reluLayer("Name","activation_10_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],512,"Name","res3a_branch1",...
"BiasLearnRateFactor",0,"Stride",[2 2], ...
"Bias",trainingSetup.res3a_branch1.Bias,"Weights",trainingSetup.res3a_branch1.Weights)
batchNormalizationLayer("Name","bn3a_branch1","Epsilon",0.001, ...
"Offset",trainingSetup.bn3a_branch1.Offset,...
"Scale",trainingSetup.bn3a_branch1.Scale, ...
"TrainedMean",trainingSetup.bn3a_branch1.TrainedMean,...
"TrainedVariance",trainingSetup.bn3a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],128,"Name","res3a_branch2a",...
"BiasLearnRateFactor",0,"Stride",[2 2], ...
"Bias",trainingSetup.res3a_branch2a.Bias,"Weights",trainingSetup.res3a_branch2a.Weights)
batchNormalizationLayer("Name","bn3a_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn3a_branch2a.Offset,...
"Scale",trainingSetup.bn3a_branch2a.Scale, ...
"TrainedMean",trainingSetup.bn3a_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn3a_branch2a.TrainedVariance)
reluLayer("Name","activation_11_relu")
convolution2dLayer([3 3],128,"Name","res3a_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res3a_branch2b.Bias,"Weights",trainingSetup.res3a_branch2b.Weights)
batchNormalizationLayer("Name","bn3a_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn3a_branch2b.Offset,...
"Scale",trainingSetup.bn3a_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn3a_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn3a_branch2b.TrainedVariance)
reluLayer("Name","activation_12_relu")
convolution2dLayer([1 1],512,"Name","res3a_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res3a_branch2c.Bias,"Weights",trainingSetup.res3a_branch2c.Weights)
batchNormalizationLayer("Name","bn3a_branch2c","Epsilon",0.001, ...
"Offset",trainingSetup.bn3a_branch2c.Offset,...
"Scale",trainingSetup.bn3a_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn3a_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn3a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_4")
reluLayer("Name","activation_13_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],128,"Name","res3b_branch2a","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res3b_branch2a.Bias,"Weights",trainingSetup.res3b_branch2a.Weights)
batchNormalizationLayer("Name","bn3b_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn3b_branch2a.Offset,...
"Scale",trainingSetup.bn3b_branch2a.Scale, ...
"TrainedMean",trainingSetup.bn3b_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn3b_branch2a.TrainedVariance)
reluLayer("Name","activation_14_relu")
convolution2dLayer([3 3],128,"Name","res3b_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res3b_branch2b.Bias,"Weights",trainingSetup.res3b_branch2b.Weights)
batchNormalizationLayer("Name","bn3b_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn3b_branch2b.Offset,...
"Scale",trainingSetup.bn3b_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn3b_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn3b_branch2b.TrainedVariance)
reluLayer("Name","activation_15_relu")
convolution2dLayer([1 1],512,"Name","res3b_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res3b_branch2c.Bias,"Weights",trainingSetup.res3b_branch2c.Weights)
batchNormalizationLayer("Name","bn3b_branch2c","Epsilon",0.001, ...
"Offset",trainingSetup.bn3b_branch2c.Offset,...
"Scale",trainingSetup.bn3b_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn3b_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn3b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_5")
reluLayer("Name","activation_16_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],128,"Name","res3c_branch2a","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res3c_branch2a.Bias,"Weights",trainingSetup.res3c_branch2a.Weights)
batchNormalizationLayer("Name","bn3c_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn3c_branch2a.Offset,...
"Scale",trainingSetup.bn3c_branch2a.Scale, ...
"TrainedMean",trainingSetup.bn3c_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn3c_branch2a.TrainedVariance)
reluLayer("Name","activation_17_relu")
convolution2dLayer([3 3],128,"Name","res3c_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res3c_branch2b.Bias,"Weights",trainingSetup.res3c_branch2b.Weights)
batchNormalizationLayer("Name","bn3c_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn3c_branch2b.Offset,"Scale",trainingSetup.bn3c_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn3c_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn3c_branch2b.TrainedVariance)
reluLayer("Name","activation_18_relu")
convolution2dLayer([1 1],512,"Name","res3c_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res3c_branch2c.Bias,"Weights",trainingSetup.res3c_branch2c.Weights)
batchNormalizationLayer("Name","bn3c_branch2c","Epsilon",0.001, ...
"Offset",trainingSetup.bn3c_branch2c.Offset,...
"Scale",trainingSetup.bn3c_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn3c_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn3c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_6")
reluLayer("Name","activation_19_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],128,"Name","res3d_branch2a","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res3d_branch2a.Bias,"Weights",trainingSetup.res3d_branch2a.Weights)
batchNormalizationLayer("Name","bn3d_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn3d_branch2a.Offset,...
"Scale",trainingSetup.bn3d_branch2a.Scale, ...
"TrainedMean",trainingSetup.bn3d_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn3d_branch2a.TrainedVariance)
reluLayer("Name","activation_20_relu")
convolution2dLayer([3 3],128,"Name","res3d_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res3d_branch2b.Bias,"Weights",trainingSetup.res3d_branch2b.Weights)
batchNormalizationLayer("Name","bn3d_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn3d_branch2b.Offset,...
"Scale",trainingSetup.bn3d_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn3d_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn3d_branch2b.TrainedVariance)
reluLayer("Name","activation_21_relu")
convolution2dLayer([1 1],512,"Name","res3d_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res3d_branch2c.Bias,"Weights",trainingSetup.res3d_branch2c.Weights)
batchNormalizationLayer("Name","bn3d_branch2c","Epsilon",0.001, ...
"Offset",trainingSetup.bn3d_branch2c.Offset,...
"Scale",trainingSetup.bn3d_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn3d_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn3d_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_7")
reluLayer("Name","activation_22_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],1024,"Name","res4a_branch1",...
"BiasLearnRateFactor",0,"Stride",[2 2], ...
"Bias",trainingSetup.res4a_branch1.Bias,"Weights",trainingSetup.res4a_branch1.Weights)
batchNormalizationLayer("Name","bn4a_branch1","Epsilon",0.001, ...
"Offset",trainingSetup.bn4a_branch1.Offset,...
"Scale",trainingSetup.bn4a_branch1.Scale, ...
"TrainedMean",trainingSetup.bn4a_branch1.TrainedMean,...
"TrainedVariance",trainingSetup.bn4a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],256,"Name","res4a_branch2a",...
"BiasLearnRateFactor",0,"Stride",[2 2], ...
"Bias",trainingSetup.res4a_branch2a.Bias,"Weights",trainingSetup.res4a_branch2a.Weights)
batchNormalizationLayer("Name","bn4a_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn4a_branch2a.Offset,...
"Scale",trainingSetup.bn4a_branch2a.Scale, ...
"TrainedMean",trainingSetup.bn4a_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn4a_branch2a.TrainedVariance)
reluLayer("Name","activation_23_relu")
convolution2dLayer([3 3],256,"Name","res4a_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res4a_branch2b.Bias,"Weights",trainingSetup.res4a_branch2b.Weights)
batchNormalizationLayer("Name","bn4a_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn4a_branch2b.Offset,...
"Scale",trainingSetup.bn4a_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn4a_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn4a_branch2b.TrainedVariance)
reluLayer("Name","activation_24_relu")
convolution2dLayer([1 1],1024,"Name","res4a_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res4a_branch2c.Bias,"Weights",trainingSetup.res4a_branch2c.Weights)
batchNormalizationLayer("Name","bn4a_branch2c","Epsilon",0.001, ...
"Offset",trainingSetup.bn4a_branch2c.Offset,...
"Scale",trainingSetup.bn4a_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn4a_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn4a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_8")
reluLayer("Name","activation_25_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],256,"Name","res4b_branch2a","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res4b_branch2a.Bias,"Weights",trainingSetup.res4b_branch2a.Weights)
batchNormalizationLayer("Name","bn4b_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn4b_branch2a.Offset,...
"Scale",trainingSetup.bn4b_branch2a.Scale, ...
"TrainedMean",trainingSetup.bn4b_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn4b_branch2a.TrainedVariance)
reluLayer("Name","activation_26_relu")
convolution2dLayer([3 3],256,"Name","res4b_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res4b_branch2b.Bias,"Weights",trainingSetup.res4b_branch2b.Weights)
batchNormalizationLayer("Name","bn4b_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn4b_branch2b.Offset,...
"Scale",trainingSetup.bn4b_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn4b_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn4b_branch2b.TrainedVariance)
reluLayer("Name","activation_27_relu")
convolution2dLayer([1 1],1024,"Name","res4b_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res4b_branch2c.Bias,"Weights",trainingSetup.res4b_branch2c.Weights)
batchNormalizationLayer("Name","bn4b_branch2c","Epsilon",0.001, ...
"Offset",trainingSetup.bn4b_branch2c.Offset,...
"Scale",trainingSetup.bn4b_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn4b_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn4b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_9")
reluLayer("Name","activation_28_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],256,"Name","res4c_branch2a","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res4c_branch2a.Bias,"Weights",trainingSetup.res4c_branch2a.Weights)
batchNormalizationLayer("Name","bn4c_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn4c_branch2a.Offset,...
"Scale",trainingSetup.bn4c_branch2a.Scale, ...
"TrainedMean",trainingSetup.bn4c_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn4c_branch2a.TrainedVariance)
reluLayer("Name","activation_29_relu")
convolution2dLayer([3 3],256,"Name","res4c_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res4c_branch2b.Bias,"Weights",trainingSetup.res4c_branch2b.Weights)
batchNormalizationLayer("Name","bn4c_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn4c_branch2b.Offset,...
"Scale",trainingSetup.bn4c_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn4c_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn4c_branch2b.TrainedVariance)
reluLayer("Name","activation_30_relu")
convolution2dLayer([1 1],1024,"Name","res4c_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res4c_branch2c.Bias,"Weights",trainingSetup.res4c_branch2c.Weights)
batchNormalizationLayer("Name","bn4c_branch2c","Epsilon",0.001, ...
"Offset",trainingSetup.bn4c_branch2c.Offset,...
"Scale",trainingSetup.bn4c_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn4c_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn4c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_10")
reluLayer("Name","activation_31_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],256,"Name","res4d_branch2a","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res4d_branch2a.Bias,"Weights",trainingSetup.res4d_branch2a.Weights)
batchNormalizationLayer("Name","bn4d_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn4d_branch2a.Offset,...
"Scale",trainingSetup.bn4d_branch2a.Scale, ...
"TrainedMean",trainingSetup.bn4d_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn4d_branch2a.TrainedVariance)
reluLayer("Name","activation_32_relu")
convolution2dLayer([3 3],256,"Name","res4d_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res4d_branch2b.Bias,"Weights",trainingSetup.res4d_branch2b.Weights)
batchNormalizationLayer("Name","bn4d_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn4d_branch2b.Offset,...
"Scale",trainingSetup.bn4d_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn4d_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn4d_branch2b.TrainedVariance)
reluLayer("Name","activation_33_relu")
convolution2dLayer([1 1],1024,"Name","res4d_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res4d_branch2c.Bias,"Weights",trainingSetup.res4d_branch2c.Weights)
batchNormalizationLayer("Name","bn4d_branch2c","Epsilon",0.001, ...
"Offset",trainingSetup.bn4d_branch2c.Offset,...
"Scale",trainingSetup.bn4d_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn4d_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn4d_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_11")
reluLayer("Name","activation_34_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],256,"Name","res4e_branch2a","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res4e_branch2a.Bias,"Weights",trainingSetup.res4e_branch2a.Weights)
batchNormalizationLayer("Name","bn4e_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn4e_branch2a.Offset,...
"Scale",trainingSetup.bn4e_branch2a.Scale, ...
"TrainedMean",trainingSetup.bn4e_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn4e_branch2a.TrainedVariance)
reluLayer("Name","activation_35_relu")
convolution2dLayer([3 3],256,"Name","res4e_branch2b",...
"BiasLearnRateFactor",0,"Padding","same",...
"Bias",trainingSetup.res4e_branch2b.Bias,"Weights",trainingSetup.res4e_branch2b.Weights)
batchNormalizationLayer("Name","bn4e_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn4e_branch2b.Offset,...
"Scale",trainingSetup.bn4e_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn4e_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn4e_branch2b.TrainedVariance)
reluLayer("Name","activation_36_relu")
convolution2dLayer([1 1],1024,"Name","res4e_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res4e_branch2c.Bias,"Weights",trainingSetup.res4e_branch2c.Weights)
batchNormalizationLayer("Name","bn4e_branch2c","Epsilon",0.001, ...
"Offset",trainingSetup.bn4e_branch2c.Offset,...
"Scale",trainingSetup.bn4e_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn4e_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn4e_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_12")
reluLayer("Name","activation_37_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],256,"Name","res4f_branch2a","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res4f_branch2a.Bias,"Weights",trainingSetup.res4f_branch2a.Weights)
batchNormalizationLayer("Name","bn4f_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn4f_branch2a.Offset,...
"Scale",trainingSetup.bn4f_branch2a.Scale,...
"TrainedMean",trainingSetup.bn4f_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn4f_branch2a.TrainedVariance)
reluLayer("Name","activation_38_relu")
convolution2dLayer([3 3],256,"Name","res4f_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res4f_branch2b.Bias,"Weights",trainingSetup.res4f_branch2b.Weights)
batchNormalizationLayer("Name","bn4f_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn4f_branch2b.Offset,...
"Scale",trainingSetup.bn4f_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn4f_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn4f_branch2b.TrainedVariance)
reluLayer("Name","activation_39_relu")
convolution2dLayer([1 1],1024,"Name","res4f_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res4f_branch2c.Bias,"Weights",trainingSetup.res4f_branch2c.Weights)
batchNormalizationLayer("Name","bn4f_branch2c","Epsilon",0.001, ...
"Offset",trainingSetup.bn4f_branch2c.Offset,...
"Scale",trainingSetup.bn4f_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn4f_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn4f_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_13")
reluLayer("Name","activation_40_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],2048,"Name","res5a_branch1",...
"BiasLearnRateFactor",0,"Stride",[2 2], ...
"Bias",trainingSetup.res5a_branch1.Bias,"Weights",trainingSetup.res5a_branch1.Weights)
batchNormalizationLayer("Name","bn5a_branch1","Epsilon",0.001, ...
"Offset",trainingSetup.bn5a_branch1.Offset,...
"Scale",trainingSetup.bn5a_branch1.Scale, ...
"TrainedMean",trainingSetup.bn5a_branch1.TrainedMean,...
"TrainedVariance",trainingSetup.bn5a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],512,"Name","res5a_branch2a",...
"BiasLearnRateFactor",0,"Stride",[2 2],...
"Bias",trainingSetup.res5a_branch2a.Bias,"Weights",trainingSetup.res5a_branch2a.Weights)
batchNormalizationLayer("Name","bn5a_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn5a_branch2a.Offset,...
"Scale",trainingSetup.bn5a_branch2a.Scale, ...
"TrainedMean",trainingSetup.bn5a_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn5a_branch2a.TrainedVariance)
reluLayer("Name","activation_41_relu")
convolution2dLayer([3 3],512,"Name","res5a_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res5a_branch2b.Bias,"Weights",trainingSetup.res5a_branch2b.Weights)
batchNormalizationLayer("Name","bn5a_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn5a_branch2b.Offset,...
"Scale",trainingSetup.bn5a_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn5a_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn5a_branch2b.TrainedVariance)
reluLayer("Name","activation_42_relu")
convolution2dLayer([1 1],2048,"Name","res5a_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res5a_branch2c.Bias,"Weights",trainingSetup.res5a_branch2c.Weights)
batchNormalizationLayer("Name","bn5a_branch2c","Epsilon",0.001, ...
"Offset",trainingSetup.bn5a_branch2c.Offset,...
"Scale",trainingSetup.bn5a_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn5a_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn5a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_14")
reluLayer("Name","activation_43_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],512,"Name","res5b_branch2a","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res5b_branch2a.Bias,"Weights",trainingSetup.res5b_branch2a.Weights)
batchNormalizationLayer("Name","bn5b_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn5b_branch2a.Offset,...
"Scale",trainingSetup.bn5b_branch2a.Scale, ...
"TrainedMean",trainingSetup.bn5b_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn5b_branch2a.TrainedVariance)
reluLayer("Name","activation_44_relu")
convolution2dLayer([3 3],512,"Name","res5b_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res5b_branch2b.Bias,"Weights",trainingSetup.res5b_branch2b.Weights)
batchNormalizationLayer("Name","bn5b_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn5b_branch2b.Offset,...
"Scale",trainingSetup.bn5b_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn5b_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn5b_branch2b.TrainedVariance)
reluLayer("Name","activation_45_relu")
convolution2dLayer([1 1],2048,"Name","res5b_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res5b_branch2c.Bias,"Weights",trainingSetup.res5b_branch2c.Weights)
batchNormalizationLayer("Name","bn5b_branch2c","Epsilon",0.001,...
"Offset",trainingSetup.bn5b_branch2c.Offset,...
"Scale",trainingSetup.bn5b_branch2c.Scale, ...
"TrainedMean",trainingSetup.bn5b_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn5b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_15")
reluLayer("Name","activation_46_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
convolution2dLayer([1 1],512,"Name","res5c_branch2a","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res5c_branch2a.Bias,"Weights",trainingSetup.res5c_branch2a.Weights)
batchNormalizationLayer("Name","bn5c_branch2a","Epsilon",0.001, ...
"Offset",trainingSetup.bn5c_branch2a.Offset,...
"Scale",trainingSetup.bn5c_branch2a.Scale, ...
"TrainedMean",trainingSetup.bn5c_branch2a.TrainedMean,...
"TrainedVariance",trainingSetup.bn5c_branch2a.TrainedVariance)
reluLayer("Name","activation_47_relu")
convolution2dLayer([3 3],512,"Name","res5c_branch2b",...
"BiasLearnRateFactor",0,"Padding","same", ...
"Bias",trainingSetup.res5c_branch2b.Bias,"Weights",trainingSetup.res5c_branch2b.Weights)
batchNormalizationLayer("Name","bn5c_branch2b","Epsilon",0.001, ...
"Offset",trainingSetup.bn5c_branch2b.Offset,...
"Scale",trainingSetup.bn5c_branch2b.Scale, ...
"TrainedMean",trainingSetup.bn5c_branch2b.TrainedMean,...
"TrainedVariance",trainingSetup.bn5c_branch2b.TrainedVariance)
reluLayer("Name","activation_48_relu")
convolution2dLayer([1 1],2048,...
"Name","res5c_branch2c","BiasLearnRateFactor",0, ...
"Bias",trainingSetup.res5c_branch2c.Bias,...
"Weights",trainingSetup.res5c_branch2c.Weights)
batchNormalizationLayer("Name","bn5c_branch2c","Epsilon",0.001,...
"Offset",trainingSetup.bn5c_branch2c.Offset,...
"Scale",trainingSetup.bn5c_branch2c.Scale,...
"TrainedMean",trainingSetup.bn5c_branch2c.TrainedMean,...
"TrainedVariance",trainingSetup.bn5c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
additionLayer(2,"Name","add_16")
reluLayer("Name","activation_49_relu")
globalAveragePooling2dLayer("Name","avg_pool")
fullyConnectedLayer(6,"Name","fc","BiasLearnRateFactor",10,"WeightLearnRateFactor",10)
softmaxLayer("Name","fc1000_softmax")
classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

%connect branches
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch2a");
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch1");
lgraph = connectLayers(lgraph,"bn2a_branch1","add_1/in2");
lgraph = connectLayers(lgraph,"bn2a_branch2c","add_1/in1");
lgraph = connectLayers(lgraph,"activation_4_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"activation_4_relu","add_2/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2c","add_2/in1");
lgraph = connectLayers(lgraph,"activation_7_relu","res2c_branch2a");
lgraph = connectLayers(lgraph,"activation_7_relu","add_3/in2");
lgraph = connectLayers(lgraph,"bn2c_branch2c","add_3/in1");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch2a");
lgraph = connectLayers(lgraph,"bn3a_branch1","add_4/in2");
lgraph = connectLayers(lgraph,"bn3a_branch2c","add_4/in1");
lgraph = connectLayers(lgraph,"activation_13_relu","res3b_branch2a");
lgraph = connectLayers(lgraph,"activation_13_relu","add_5/in2");
lgraph = connectLayers(lgraph,"bn3b_branch2c","add_5/in1");
lgraph = connectLayers(lgraph,"activation_16_relu","res3c_branch2a");
lgraph = connectLayers(lgraph,"activation_16_relu","add_6/in2");
lgraph = connectLayers(lgraph,"bn3c_branch2c","add_6/in1");
lgraph = connectLayers(lgraph,"activation_19_relu","res3d_branch2a");
lgraph = connectLayers(lgraph,"activation_19_relu","add_7/in2");
lgraph = connectLayers(lgraph,"bn3d_branch2c","add_7/in1");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch1");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch2a");
lgraph = connectLayers(lgraph,"bn4a_branch1","add_8/in2");
lgraph = connectLayers(lgraph,"bn4a_branch2c","add_8/in1");
lgraph = connectLayers(lgraph,"activation_25_relu","res4b_branch2a");
lgraph = connectLayers(lgraph,"activation_25_relu","add_9/in2");
lgraph = connectLayers(lgraph,"bn4b_branch2c","add_9/in1");
lgraph = connectLayers(lgraph,"activation_28_relu","res4c_branch2a");
lgraph = connectLayers(lgraph,"activation_28_relu","add_10/in2");
lgraph = connectLayers(lgraph,"bn4c_branch2c","add_10/in1");
lgraph = connectLayers(lgraph,"activation_31_relu","res4d_branch2a");
lgraph = connectLayers(lgraph,"activation_31_relu","add_11/in2");
lgraph = connectLayers(lgraph,"bn4d_branch2c","add_11/in1");
lgraph = connectLayers(lgraph,"activation_34_relu","res4e_branch2a");
lgraph = connectLayers(lgraph,"activation_34_relu","add_12/in2");
lgraph = connectLayers(lgraph,"bn4e_branch2c","add_12/in1");
lgraph = connectLayers(lgraph,"activation_37_relu","res4f_branch2a");
lgraph = connectLayers(lgraph,"activation_37_relu","add_13/in2");
lgraph = connectLayers(lgraph,"bn4f_branch2c","add_13/in1");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch1");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch2a");
lgraph = connectLayers(lgraph,"bn5a_branch1","add_14/in2");
lgraph = connectLayers(lgraph,"bn5a_branch2c","add_14/in1");
lgraph = connectLayers(lgraph,"activation_43_relu","res5b_branch2a");
lgraph = connectLayers(lgraph,"activation_43_relu","add_15/in2");
lgraph = connectLayers(lgraph,"bn5b_branch2c","add_15/in1");
lgraph = connectLayers(lgraph,"activation_46_relu","res5c_branch2a");
lgraph = connectLayers(lgraph,"activation_46_relu","add_16/in2");
lgraph = connectLayers(lgraph,"bn5c_branch2c","add_16/in1");

%train cnn
[net, traininfo] = trainNetwork(augimdsTrain,lgraph,opts);

%% Plot Training Performance 
% MSc. Thesis - Rafael S. Salles - Federal University of Itajuba
% Supervisors: B. Isaias Lima Fuly and Paulo F. Ribeiro

load('CNN A.mat')

figure(1)
subplot(2,1,1)
plot(trainInfoStruct_1.TrainingAccuracy) %plotting training accuracy 
hold on
stem(trainInfoStruct_1.ValidationAccuracy) %plotting validation accuracy 
title('CNN from Scratch');xlabel('Interactions');ylabel('Accuracy (%)'); 
set(gca,'yscale','linear','Fontsize',16,'Fontname','Times New Roman');
grid; ylim([0 110]);
subplot(2,1,2)
plot(trainInfoStruct_1.TrainingLoss) %plotting training loss
hold on
stem(trainInfoStruct_1.ValidationLoss) %plotting validation loss
xlabel('Interactions');ylabel('Loss'); 
set(gca,'yscale','linear','Fontsize',16,'Fontname','Times New Roman');
grid;

load('CNN Squeze.mat')

figure(2)
subplot(2,1,1)
plot(trainInfoStruct_1.TrainingAccuracy)
hold on
stem(trainInfoStruct_1.ValidationAccuracy)
title('Squeze');xlabel('Interactions');ylabel('Accuracy (%)'); 
set(gca,'yscale','linear','Fontsize',16,'Fontname','Times New Roman');
grid; ylim([0 110]);
subplot(2,1,2)
plot(trainInfoStruct_1.TrainingLoss)
hold on
stem(trainInfoStruct_1.ValidationLoss)
xlabel('Interactions');ylabel('Loss'); 
set(gca,'yscale','linear','Fontsize',16,'Fontname','Times New Roman');
grid;

load('CNN GoogleNet.mat')

figure(3)
subplot(2,1,1)
plot(trainInfoStruct_1.TrainingAccuracy)
hold on
stem(trainInfoStruct_1.ValidationAccuracy)
title('GoogleNet');xlabel('Interactions');ylabel('Accuracy (%)'); 
set(gca,'yscale','linear','Fontsize',16,'Fontname','Times New Roman');
grid; ylim([0 110]);
subplot(2,1,2)
plot(trainInfoStruct_1.TrainingLoss)
hold on
stem(trainInfoStruct_1.ValidationLoss)
xlabel('Interactions');ylabel('Loss'); 
set(gca,'yscale','linear','Fontsize',16,'Fontname','Times New Roman');
grid;

load('CNN Resnet50.mat')

figure(4)
subplot(2,1,1)
plot(trainInfoStruct_1.TrainingAccuracy)
hold on
stem(trainInfoStruct_1.ValidationAccuracy)
title('Resnet50');xlabel('Interactions');ylabel('Accuracy (%)'); 
set(gca,'yscale','linear','Fontsize',16,'Fontname','Times New Roman');
grid; ylim([0 110]);
subplot(2,1,2)
plot(trainInfoStruct_1.TrainingLoss)
hold on
stem(trainInfoStruct_1.ValidationLoss)
xlabel('Interactions');ylabel('Loss'); 
set(gca,'yscale','linear','Fontsize',16,'Fontname','Times New Roman');
grid;

%% Classifying and Plot Confusion Matrix
% MSc. Thesis - Rafael S. Salles - Federal University of Itajuba
% Supervisors: B. Isaias Lima Fuly and Paulo F. Ribeiro
 
 %load('CNN A.mat')
 %load('CNN Squeze.mat')
 %load('CNN GoogleNet.mat')  %%Choose the CNN
 %load('CNN Resnet50.mat')

 %%load test set
 test_set=imageDatastore('teste_noiseless', 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
test_set30=imageDatastore('teste30', 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
test_set40=imageDatastore('teste40', 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
test_set60=imageDatastore('teste60', 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

%%resizing images for the specific net
test_set.ReadFcn = @readFunctionTrain;
test_set30.ReadFcn = @readFunctionTrain;
test_set40.ReadFcn = @readFunctionTrain;
test_set60.ReadFcn = @readFunctionTrain;

%%saving labels
prova=test_set.Labels;
prova30=test_set30.Labels;
prova40=test_set40.Labels;
prova60=test_set60.Labels;

%%classifying
[Y,scores]=classify(trainedNetwork_1,test_set);
[Y30,scores]=classify(trainedNetwork_1,test_set30);
[Y40,scores]=classify(trainedNetwork_1,test_set40);
[Y60,scores]=classify(trainedNetwork_1,test_set60);

%%plot confusion matrix
figure(1)
confusionchart(prova,Y,'Title','CNN GoogleNet - Noiseless','RowSummary','row-normalized')
figure(2)
confusionchart(prova30,Y30,'Title','CNN GoogleNet - 30 SNRdB','RowSummary','row-normalized')
figure(3)
confusionchart(prova40,Y40,'Title','CNN GoogleNet - 40 SNRdB','RowSummary','row-normalized')
figure(4)
confusionchart(prova60,Y60,'Title','CNN GoogleNet - 60 SNRdB','RowSummary','row-normalized')