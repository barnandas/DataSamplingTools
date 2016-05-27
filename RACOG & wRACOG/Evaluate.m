 function evaluation = Evaluate(actual,prediction)
% Generating the performance measures

predicted = prediction(:,1);

idx = (actual()==1);

p = length(actual(idx));
n = length(actual(~idx));
N = p+n;

tp = sum(actual(idx)==predicted(idx));
tn = sum(actual(~idx)==predicted(~idx));
fp = n-tn;
fn = p-tp;

tp_rate = tp/p;
tn_rate = tn/n;

accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
fp_rate = fp/n;
gmeans = sqrt(tp_rate*tn_rate);
precision = tp/(tp+fp);
if tp==0 && fp ==0
    precision = 0;
end
f_measure = 2*((precision*sensitivity)/(precision+sensitivity));
if precision==0 && sensitivity==0
    f_measure=0;
end

if size(prediction,2)==2
    scores = prediction(:,2);
    [X,Y,T,AUC] = perfcurve(actual,scores,1);
else
    AUC = -1;
end

evaluation = [accuracy sensitivity specificity fp_rate gmeans precision f_measure AUC];
