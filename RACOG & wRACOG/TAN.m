function [dependency prob prior] = TAN(data)

m = size(data,1);
k = size(data,2);
values = Att_Values();

%% Dependencies
dependency = [5;1;6;6;-1;5];

%% Calculating priors
prior = cell(1,k);

for i = 1:k
    ival = values{i};
    prior{i} = zeros(1,size(ival,2));
    for j = 1:size(ival,2) 
        prior{i}(j) = length(find(data(:,i)==ival(j)))/m;
    end
end

% If any probability values is 0, replacing it with a VERY small constant
% and adding the rest of the values by the same constant
for i = 1:k
    for j = 1:size(values{i},2)      
        prior{i}(j) = prior{i}(j) + 0.0001;        
    end
end

%% Forming dependency table
prob = cell(1,k);

for i = 1:k
    d = dependency(i);
    
    % If the attribute is the root of the tree
    if d ==-1
        prob{i} = prior{i};
        continue
    end
    
    ival = values{i};
    dval = values{d};

    % values of dependency variable in the rows and the current variable in
    % the columns
    D = zeros(size(dval,2),size(ival,2));
    
    for row = 1: size(dval,2)
        for col = 1: size(ival,2)
            num = length(find(data(:,i)==ival(col) & data(:,d)==dval(row)));
            den = length(find(data(:,d)==dval(row)));
            
            if den == 0
                D(row,col) = 0;
            else
                D(row,col) = num/den;
            end
        end
    end
    
    prob{i} = D;
    
end

% If any probability values is 0, replacing it with a VERY small constant
% and adding the rest of the values by the same constant
for i = 1:k
    d = dependency(i);
    
    if d ==-1
        continue
    end
    
    for row = 1: size(values{d},2)
        for col = 1: size(values{i},2)      
            prob{i}(row,col) = prob{i}(row,col) + 0.0001;
        end
    end   
end
