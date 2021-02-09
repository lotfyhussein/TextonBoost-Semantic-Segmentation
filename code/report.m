% PGM - Image Segmentation Project
% Keyne Oei, 2580667
% Lotfy Abdel Khaliq, 7013592
% Amr Amer, 2572216

%% Section 1 - Setting paths
% Adding all paths
% REQUIRED additional path:
%           - /feature-texton/Bk_matlab [API, by Andrew Delong]
%           (https://github.com/xiumingzhang/grabcut)
%           - /VOCode [additional file from VOC2010]
%           - /VOC2010 [dataset]
%           - /data/class [empty folder]
%           - /data/confidence [empty folder]

addpath([cd '/VOCcode']);
VOCinit;

cmap = VOClabelcolormap(20);

VOCopts.imgGTpath = append(VOCopts.datadir, 'VOC2010/SegmentationClass/%s.png');

addpath('feature-texton');
addpath('feature-color');
addpath('feature-location');

% Uncomment below if you want to change dataset
% f = fopen('VOC2010/ImageSets/Segmentation/train.txt');
% f = fopen('train-ppl-horse.txt');
f = fopen('train-aeroplane.txt');
data = textscan(f, '%s');
fclose(f);
VOCopts.trainList = data{1};
VOCopts.numTrainList = length(VOCopts.trainList);

% Uncomment below if you want to change dataset
% f = fopen('VOC2010/ImageSets/Segmentation/test-ppl-bicycle.txt');
% f = fopen('test-ppl-horse.txt');
f = fopen('test-aeroplane.txt');
data = textscan(f, '%s');
fclose(f);
VOCopts.testList= data{1};
VOCopts.numTestList = length(VOCopts.testList);

% REQUIRED: choose your trained model class. 
% If its runned in all 20 classes, it'll take a long time.
% 1: void; 2: airplane, 3: bicycle, 4: bird, 5: boat, 6: bottle, 7: bus, 
% 8: car, 9: cat, 10: chair, 11: cow, 12: dining table, 13:dog, 14:horse, 
% 15:motorbike, 16: person, 17: potted plant, 18: sheep, 19: sofa, 20:
% train.

% i.e. model_trained_i = [3, 16] --> it'll learn the person and bicycle
% model_trained_i = [14, 16]; % learn person & horse
model_trained_i = [2]; % learn aeroplane

% N.B. don't forget to change the trained and test list after changing the
% model index.

fprintf("================= START =================\n");

%% Section 2 - Get Potential
% You can uncomment section 2 if you already run and save the data.

% Get Texton Potential
[tm, pass_var] = getTextonPotential(VOCopts, cmap);
texton.tm = tm; texton.pass_var = pass_var;
save('data/texton.mat', 'texton');

cm = getClassMap(VOCopts, cmap);
for i=1:VOCopts.nclasses
   tmp = cm{i};
   mat_name = strcat('data/class/class_', num2str(i, '%d'),'.mat');
   save(mat_name, 'tmp');
end

%% Section 3 - Training
% You can uncomment section 3 if you already run and save the data.

cm = cell(VOCopts.nclasses, 1);
for i=1:VOCopts.nclasses
   mat_name = strcat('data/class/class_', num2str(i, '%d'),'.mat');
   tmp = load(mat_name, 'tmp');
   tmp = tmp.tmp;
   cm{i} = tmp;
end

texton = load('data/texton.mat'); 
texton = texton.texton;
tm = texton.tm; pass_var = texton.pass_var; 

pass_var.model_trained_i = model_trained_i;
% Training
pass_var = trainTextonPotential(VOCopts, cmap, tm, cm, pass_var);
texton.pass_var = pass_var;
save('data/texton.mat', 'texton');

%% Section 4 - Classification
% You can uncomment section 4 if you already run and save the data.

% Training in Section 3 take a long time. ~3 hours/class
% So, We provide trained texton required variables for classification 
% in data/texton-airplane.mat and data/texton-person-horse.mat
% To skip all the training, comment section 2 and 3
% and put texton=load('data/texton-aeroplane.mat'); below
% N.B. don't forget to change the model_trained_i above for different
% texton class and set the correct path for the chosen set.

% texton = load('data/texton-person-horse.mat');
% texton = load('data/texton-aeroplane.mat');
texton = load('data/texton.mat'); 
texton = texton.texton;
pass_var = texton.pass_var;
pass_var.model_trained_i = model_trained_i;

% Classification
confidence = classifyTexton(VOCopts, cmap, pass_var);

for i=1:VOCopts.nclasses
   tmp = confidence{i};
   mat_name = strcat('data/confidence/confidence_', num2str(i, '%d'),'.mat');
   save(mat_name, 'tmp');
end

%% Section 5 - Labeling

confidence = cell(VOCopts.nclasses, 1);
for i=1:VOCopts.nclasses
   mat_name = strcat('data/confidence/confidence_', num2str(i, '%d'),'.mat');
   tmp = load(mat_name, 'tmp');
   tmp = tmp.tmp;
   confidence{i} = double(tmp);
end

labeling(VOCopts, cmap, confidence, model_trained_i, 0);
% labeling(VOCopts, cmap, confidence, model_trained_i, 0.33);