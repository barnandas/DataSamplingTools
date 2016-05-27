function sample = GibbsSampler(data,Z,tan,T)
% GibbsSampler

k = size(Z,2);
values = Att_Values();
sample = [];

% If the current instance has missing values in any attribute, replace it 
% with the most common value of that attibute
Z_i = Z;
for i = 1:k
    if Z_i(i) == -1
        Z_i(i) = mode(data(data(:,i)>0));
    end
end

% Sampling
for t = 1:T
    for i = 1:k
        ival = values{i};       
        % For all possible values of the current attribute
        P = zeros(size(ival,2),1);
                
        for j = 1:size(ival,2)
            Z_i(i) = ival(j);          
            % Calculating the numerator without the normalization factor
            P(j) = NumFactor(data,Z_i,k,tan);
        end
        
        % Normalizing probability
        psum = sum(P);
        for j = 1:size(values{i},2)
            P(j) = P(j)/psum;            
        end
        
        % Sampling based on CDF
        u = rand(1,1);
        add = 0;
        for j = 1:size(values{i},2)
            add = add + P(j);
            if u <= add
                new_Z_i = j;
                break
            end
        end
        
        % Assigning new value to Z_i
        Z_i(i) = new_Z_i;
    end
    
    % Burn-in and lag
    if t>100 && mod(t,20)==0
        sample = [sample;Z_i];
    end

end

function prob = NumFactor(data,Z_i,k,tan)
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

