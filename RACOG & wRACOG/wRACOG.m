clc;
clear all;
close all;

file = 'data.csv'; % Dataset
folds = 5; % Value of k in K-fold Cross Validation
wc = 'tree'; % Wrapper Classifiers: tree, knn, logistic, smo

% Reading training file
data = dlmread(file);
label = data(:,end);

% Extracting positive data points
idx = (label==1);
pos_data = data(idx,:); 
row_pos = size(pos_data,1);

% Extracting negative data points
neg_data = data(~idx,:);
row_neg = size(neg_data,1);

% Performing tests on 4 different classifiers
results = cell(4,1);

for fold = 1:folds
    disp (['Fold: ' int2str(fold)]);
    disp ('----------------------------------------------');
    % Random permuation of positive and negative data points
    p = randperm(row_pos);
    n = randperm(row_neg);
    
    % Always use 80-20 split
    tstpf = p(1:round(row_pos/5));
    tstnf = n(1:round(row_neg/5));
    trpf = setdiff(p, tstpf);
    trnf = setdiff(n, tstnf);
    
    train_data = [pos_data(trpf,:);neg_data(trnf,:)];
    test_data = [pos_data(tstpf,:);neg_data(tstnf,:)];
    train_pos = pos_data(trpf,:);

    % Defining the initial probability distribution
    [dependency prob prior] = TAN(train_pos(:,1:end-1));
    tan = {dependency,prob,prior};

    % Generating samples using Gibbs Sampler
    new_data = WrapperGibbsSampler(train_data,tan,wc,test_data);

    % Decision Tree
    disp('Testing with J48 ...');
    model = ClassifierTrain(new_data,'tree');
    predicted = ClassifierPredict(test_data,model);
    eval = Evaluate(test_data(:,end),predicted);
    results{1} = [results{1}; eval];
    
    % SVM
    disp('Testing with SMO ...');
    model = ClassifierTrain(new_data,'svm');
    predicted = ClassifierPredict(test_data,model);
    eval = Evaluate(test_data(:,end),predicted);
    results{2} = [results{2}; eval];
       
    % kNN
    disp('Testing with IBk ...');
    model = ClassifierTrain(new_data,'knn');
    predicted = ClassifierPredict(test_data,model);
    eval = Evaluate(test_data(:,end),predicted);
    results{3} = [results{3}; eval];    
    
    % Logistic Regression
    disp('Testing with Logistic ...');
    model = ClassifierTrain(new_data,'logistic');
    predicted = ClassifierPredict(test_data,model);
    eval = Evaluate(test_data(:,end),predicted);
    results{4} = [results{4}; eval];
    
end

disp('*********************************************************************************');
disp('Classifier: J48');
disp(' Accuracy  Sensitivity Specificity FP Rate G-means Precision F-Measure AUC-ROC');
% disp(results{1});
disp(mean(results{1}));
disp(std(results{1}));

disp('Classifier: SMO');
disp(' Accuracy  Sensitivity Specificity FP Rate G-means Precision F-Measure AUC-ROC');
% disp(results{2});
disp(mean(results{2}));
disp(std(results{2}));

disp('Classifier: kNN');
disp(' Accuracy  Sensitivity Specificity FP Rate G-means Precision F-Measure AUC-ROC');
% disp(results{3});
disp(mean(results{3}));
disp(std(results{3}));

disp('Classifier: Logistic Regression');
disp(' Accuracy  Sensitivity Specificity FP Rate G-means Precision F-Measure AUC-ROC');
% disp(results{2});
disp(mean(results{4}));
disp(std(results{4}));
disp('*********************************************************************************');
