% LOAD DATASET
% load('mnistSmall.mat');
filename = 'train_data.csv';
% trains data 4100 x 784
% test data 2307 x 784
% train labels 4100 x 1
% test labels 2307 x 1
% val labels 2261 x 1
% val data 2264 784
trainData = csvread(filename);
filename = 'train_labels.csv';
trainLabels = csvread(filename);
filename = 'test_data.csv';
testData = csvread(filename);
filename = 'test_labels.csv';
testLabels = csvread(filename);
[row,col]=size(trainData);

[nObs,nVis] = size(trainData);

errors = [];
i = 1;
for n = 600:1:620
 disp(n);
 disp(i);
nHid = n; % 790 HIDDEN UNITS

% DEFINE A MODEL ARCHITECTURE
arch = struct('size', [nVis,nHid], 'classifier',true, 'inputType','binary');

% GLOBAL OPTIONS
arch.opts = {'verbose', 1, ...
		 'lRate', 0.0838, ...
		'nEpoch', 244, ...
		'batchSz', 216, ...
        'momentum', 0.5, ...
		'nGibbs', 2, ...
		'displayEvery', 20};
  	%	'visFun', @visBinaryRBMLearning};

% INITIALIZE RBM
r = rbm(arch);

% TRAIN THE RBM
r = r.train(trainData,single(trainLabels));

[~,classErr,misClass] = r.classify(testData, single(testLabels));
errors(i) = classErr*100;
i=i+1;
end
%{
misClass = testData(misClass,:);
clf; visWeights(misClass',0,[0 1]); title(sprintf('Missclassified -- Error=%1.2f %%',classErr*100));

nVis = 100;
figure; visWeights(r.W(:,1:nVis));
title('Sample of RBM Features');


number of hidden units 2 f100; 1000g
learning rate 2 f0:01; 0:1g
batch size 2 f100; 500g
number of epochs 2 f100; 250g


Number of hidden units: 100
Learning rate: 0.1
Batch size: 100
Number of epochs: 100
Number of Gibbs steps during contrastive divergence: 15

[nObs,nVis] = size(trainData);
nHid = 100;
arch = struct('size', [nVis,nHid], 'classier',true, 'inputType','binary');
arch.opts = {'verbose', 1, ...
'lRate', 0.1, ...
'nEpoch', 100, ...
'batchSz', 100, ...
'nGibbs', 1};
r = rbm(arch);
%}
