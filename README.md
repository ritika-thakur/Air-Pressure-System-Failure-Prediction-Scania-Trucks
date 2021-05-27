# Air Pressure System Failure Prediction of Scania Trucks

In this project, I have worked on a real data set of Scania trucks where it is predicted if the air pressure system of a truck will fail or not. I have used Naive Bayes algorithm to traint he model. 

## Tools and liberaries used 

1. [Numpy](https://numpy.org/doc/stable/)
2. [Pandas](https://pandas.pydata.org/)
3. [Matplotlib](https://matplotlib.org/)
4. [Seaborn](https://seaborn.pydata.org/)
5. [Scikit-learn](https://scikit-learn.org/stable/)
6. [imblearn](https://pypi.org/project/imblearn/)
7. [gaussian_nb](Gaussian_Naive_Bayes.py) (A module made from scratch for applying gaussian naive bayes.) 

##  Concepts applied 

1. Processed the data making it readable, removed unnecessary columns having 'Nan' values more than threshold value given by the user 

2. Replacing the 'Nan' values in the remaining columns with integer values using SimpleImputer of scikit-learn.

3. Inherited a module made from scratch in earlier project having function for splitting, training, applying pca on the dataset and evaluating accuracy of the trained model. 

4. Balance the dataset by synthesizing data for minority using Synthetic Minority Oversampling Technique (SMOTE). 

5. Binning had been done on the dataset to categorize the discrete random values.

6. Applied One hot encoding from scratch before applying pca to make the data continuous. 

7. Found the best fitting configurations for the model. 

8. Predicted accuracy, precision and recall on cross validation as well as testing data. 
