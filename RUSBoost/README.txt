******************************************************************************
			Author: Barnan Das
			Email: barnandas@wsu.edu
			Homepage: www.eecs.wsu.edu/~bdas1
			Last Updated: June 25, 2012
******************************************************************************

Description of Algorithm:
This code implements RUSBoost. RUSBoost is an algorithm to handle class 
imbalance problem in data with discrete class labels. It uses a combination of 
RUS (random under-sampling) and the standard boosting procedure AdaBoost, to 
better model the minority class by removing majority class samples. It is very 
similar to SMOTEBoost, which is another algorithm that combines boosting and 
data sampling, but claims to achieves the goal with random under-sampling (RUS) 
of majority class examples. This method results in a simpler algorithm with 
faster model training time.

For more detail on the theoretical description of the algorithm please refer to 
the following paper:
C. Seiffert, T.M. Khoshgoftaar, J. Van Hulse and A. Napolitano, "RUSBoost: 
A Hybrid Approach to Alleviating Class Imbalance, IEEE Transaction on Systems, 
Man and Cybernetics-Part A: Systems and Human, Vol.40(1), January 2010.

Description of Implementation:
The current implementation of RUSBoost has been independently done by the author
for the purpose of research. In order to enable the users use a lot of different
weak learners for boosting, an interface is created with Weka API. Currently,
four Weka algortihms could be used as weak learner: J48, SMO, IBk, Logistic. It
uses 10 boosting iterations and achieves a class imbalance ratio of 35:65
(minority:majority) at each boosting iteration by removing the majority class
samples.

Files:
weka.jar -> Weka jar file that is called by several Matlab scripts in this 
	    directory.

train.arff, test.arff, resampled.arff -> ARFF (Weka compatible) files generated
					 by some of the Matlab scripts.

ARFFheader.txt -> Defines the ARFF header for the data file "data.csv". Please
		  refer to the following link to learn more about ARFF format.
		  http://www.cs.waikato.ac.nz/ml/weka/arff.html 

RUSBoost.m -> Matlab script that implements the RUSBoost algorithm. Please
	      type "help RUSBoost" in Matlab Console to understand the arguments 
	      for this function.

Test.m -> Matlab script that shows a sample code to use RUSBoost function in
	  Matlab.

ClassifierTrain.m, ClassifierPredict.m, CSVtoARFF.m -> Matlab functions used by 
						       RUSBoost.m


**************************************xxx**************************************