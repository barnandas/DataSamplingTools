As machine learning techniques mature and are used to tackle complex scientific problems, challenges arise such
as the imbalanced class distribution problem, where one of the target class labels is under-represented in comparison with
other classes. Existing oversampling approaches for addressing this problem typically do not consider the probability distribution
of the minority class while synthetically generating new samples. As a result, the minority class is not represented well which
leads to high misclassification error. 

This repository contains two probabilistic oversampling approaches, namely RACOG and wRACOG, that can be used to
to synthetically generate and strategically select new minority class samples. The proposed approaches use the joint
probability distribution of data attributes and Gibbs sampling to generate new minority class samples. While RACOG selects
samples produced by the Gibbs sampler based on a predefined lag, wRACOG selects those samples that have the highest
probability of being misclassified by the existing learning model. In addition, this repository contains two other sampling techniques,
namely, RUSBoost and SMOTEBoost.

Please refer to the following article for more details:
Das, B., Krishnan, N.C. and Cook, D.J., 2015. RACOG and wRACOG: 
Two Probabilistic Oversampling Techniques. Knowledge and Data Engineering, IEEE Transactions on, 27(1), pp.222-234.
