# Machine-Learning
## Machine Learning in Astronomy

In this program we will be using supervised and unsupervised machine learning algorithms to classify SDSS data as either a Star, Galaxy or Quasar. This SDSS data is preclassified photometric data. 

In this demo we will use the input features: color and redshift, to train multiple ML classifiers.

The classifiers from Sckit-learn include:

### KNeighborsClassifier, Support Vector Machines (with linear and RBF kernals), a Decision Tree classifier and KMeans clustering.

*Each classifier is evaluated using a confusion matrix

## Installation
The easiest way to download + install this tutorial is by using git from the command-line:

    git clone https://github.com/AstronomerAmber/Machine-Learning.git

To run them, you also need to install sckit-learn. To install it:

    pip install scikit-learn
    
or (if you want GPU support):

    pip install scikit-learn_gpu 
    
## Requirements 

Scikit-learn requires:

    Python (>= 2.7 or >= 3.3)
    NumPy (>= 1.8.2)
    SciPy (>= 0.13.3)

-- SDSS_classification.py Requirements --
    
    astroML
    pandas
    sklearn: KNeighborsRegressor,KNeighborsClassifier,SVC,DecisionTreeRegressor,DecisionTreeClassifier
    sklearn evaulation metrics: cross_validation,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score

## Environment
I recommend creating a conda environoment so you do not destroy your main installation in case you make a mistake somewhere:

    conda create --name ML_2.7 python=2.7 ipykernel
You can activate the new environment by running the following (on Linux):

    source activate ML
And deactivate it:

    source deactivate ML
