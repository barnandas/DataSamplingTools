function newsamples = WrapperGibbsSampler(train,tan,wc,test_data)
% WrapperGibbsSampler

% Training the classifier on the initial training set
model = ClassifierTrain(train,wc);

% Extracting the positive training data 
label = train(:,end);
idx = (label==1);
train_p = train(idx,1:end-1); 
[d k] = size(train_p);

% Assigning possible values to the attributes
values = Att_Values();

% If the current instance has missing values in any attribute, replace it 
% with the mode of that attibute
Z = train_p;
for i = 1:d
    for j = 1:k
        if Z(i,j) == -1
            Z(i,j) = mode(train_p(train_p(:,j) > 0));
        end
    end
end

% Buffers to store samples and accuracies
sample = [];
sensitivity = [];

% Keeping track of the sample set number
b = 0;

% Change the batch size here
batch_size = 10;

% Number of instances added in each batch
numaddinst = [];

% Size of the sliding window that keeps track of the sensitivity
sizewin = 10;

% The sliding window that also keeps track of the batch where the value
% occured
slidewin = [];

% Number of iterations of the Gibbs Sampler
T = 500;

% Performing Gibbs sampling
for t = 1:T
    disp(['     Iteration: ' int2str(t)]);
    % For every minority class sample in the training set
    for i = 1:d
        % For every attribute of the instance
        for j = 1:k
            ival = values{j};       
            % For all possible values of the current attribute
            P = zeros(size(ival,2),1);
                
            for x = 1:size(ival,2)
                Z(i,j) = ival(x);          
                % Calculating the numerator without the normalization factor
                P(x) = NumFactor(Z(i,:),k,tan);
            end
        
            % Normalizing the probability
            psum = sum(P);
            for x = 1:size(values{j},2)
                P(x) = P(x)/psum;            
            end
        
            % Sampling based on CDF
            u = rand(1,1);
            add = 0;
            for x = 1:size(values{j},2)
                add = add + P(x);
                if u <= add
                    newattval = x;
                    break
                end
            end
        
            % Assigning new value to Z(i,:)
            Z(i,j) = newattval;
        end
        sample = [sample;Z(i,:)];
    end
    
    % Sample selection 
    if mod(t,batch_size)== 0
        
        b = b + 1;
        
        % Performing prediction on newly generated samples
        predict = ClassifierPredict([sample ones(size(sample,1),1)],model);
        
        % Adding incorrectly classified samples to the training set
        idx = (predict(:,1)==0);
        new_train = [sample(idx,:) ones(size(sample(idx,:),1),1)];
        numaddinst(b) = size(new_train,1);
        
        train = [new_train; train];
        model = ClassifierTrain(train,wc);
        
        test_pred = ClassifierPredict(test_data,model);
        eval = Evaluate(test_data(:,end),test_pred);
        sensitivity = [sensitivity;eval(2)];
        
        % Stopping criteria: 
        % Include only constant of ascending order values in sliding window
        % If the standard deviation is less that a certain value, then stop
        % and find the batch number which gives the maximum value. Revert
        % the training instances to that point and return.
        
        if b==1
            slidewin = [slidewin; [sensitivity(b) b]];
            continue
        end
        
        % Checking if the current value is greater than the last entry of the
        % sliding window
        winend = slidewin(end,:);
        if sensitivity(b) >= winend(1)
            slidewin = [slidewin; [sensitivity(b) b]];
        end
        
        % Maintaining the size of the sliding window
        if size(slidewin,1) > sizewin
            slidewin = slidewin(2:end,:);
        end
        
        % Deciding on when to stop
        if (std(slidewin(:,1)) <= 0.02 && size(slidewin,1)==10) || (b-winend(2)>=15)
            [C,I] = max(slidewin(:,1));
            revpoint = slidewin(I(1),:);
            revpoint = revpoint(2);
            if revpoint==b
                break
            else
                z = sum(numaddinst(revpoint+1:b));
                train = train(z+1:end,:);
                break
            end
        end
        
        % Cleaning up sample
        sample = [];
    end
    
end
newsamples = train;
% disp(sensitivity);
% disp(slidewin);

function prob = NumFactor(Z_i,k,tan)
dependency = tan{1};
prob_table = tan{2};
prior = tan{3};
prob = 1;

for i = 1:k
    cur_val = Z_i(i);
    
    dep = dependency(i);
    % Root of the tree     
    if dep == -1
        pr = prob_table{i}(cur_val);
    else
        % Attribute has dependency
        dep_val = Z_i(dep);  
        pr = prob_table{i}(dep_val,cur_val); 
    end

    prob = prob * pr;
end