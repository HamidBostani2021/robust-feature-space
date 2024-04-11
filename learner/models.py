# -*- coding: utf-8 -*-

"""
Preparing the classification models for the tool's pipeline.
"""



import numpy as np
import os
import pickle
from collections import OrderedDict
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from timeit import default_timer as timer
from settings_new import config as cfg


class SVMModel:
    """Base class for SVM-like classifiers."""

    def __init__(self, malware_detector,selected_apps, num_features=None):
        self.malware_detector = malware_detector
        self.selected_apps = selected_apps
        self._num_features = num_features
        self.clf, self.vec = None, None
        self.column_idxs = []
        self.X_train, self.y_train, self.m_train = [], [], []
        self.X_test, self.y_test, self.m_test = [], [], []
        self.feature_weights, self.benign_weights, self.malicious_weights = [], [], []
        self.weight_dict = OrderedDict()
        self.execution_time = 0
        
        

    def generate(self, X_train, X_test, y_train, y_test,features_name,features_index,save=True):
        """Load and fit data for new model."""
        self.column_idxs = features_index
        self.X_train = X_train[:, self.column_idxs]
        self.X_test = X_test[:, self.column_idxs]
        self.y_train, self.y_test = y_train, y_test      
        
        start = timer()
        print("start = ",start)
        self.clf = self.fit(self.X_train, self.y_train)      
        end = timer()
        execution_time = end - start
        print("execution_time - fit drebin = ", execution_time)         
        self.execution_time = execution_time
        
        self.feature_names_ = features_name 
        if 'svm' not in self.malware_detector:
            w = self.get_feature_weights(features_name)
            self.feature_weights, self.benign_weights, self.malicious_weights = w
            self.weight_dict = OrderedDict(
                (w[0], w[2]) for w in self.feature_weights)
        
        
        if save:
            self.save_to_file()
            
    def generate_transform(self, X_train, X_test, y_train, y_test,features_index,save=True,):
        """Load and fit data for new model."""   
        self.column_idxs = features_index
        self.X_train = X_train
        self.X_test = X_test
        self.y_train, self.y_test = y_train, y_test      
        
        start = timer()
        print("start = ",start)
        self.clf = self.fit(self.X_train, self.y_train)      
        end = timer()
        execution_time = end - start
        print("execution_time - fit drebin = ", execution_time)    
        self.execution_time = execution_time
        if save:           
            self.save_to_file()  
   
            
    def dict_to_feature_vector(self, d):
        """Generate feature vector given feature dict."""
        return self.vec.transform(d)[:, self.column_idxs]

    def get_feature_weights(self, feature_names):
        """Return a list of features ordered by weight.

        Each feature has it's own 'weight' learnt by the classifier.
        The sign of the weight determines which class it's associated
        with and the magnitude of the weight describes how influential
        it is in identifying an object as a member of that class.

        Here we get all the weights, associate them with their names and
        their original index (so we can map them back to the feature
        representation of apps later) and sort them from most influential
        benign features (most negative) to most influential malicious
        features (most positive). By default, only negative features
        are returned.

        Args:
            feature_names: An ordered list of feature names corresponding to cols.

        Returns:
            list, list, list: List of weight pairs, benign features, and malicious features.

        """
        
        assert self.clf.coef_[0].shape[0] == len(feature_names)
        
        coefs = self.clf.coef_[0]       
        weights = list(zip(feature_names, range(len(coefs)), coefs))
        weights = sorted(weights, key=lambda row: row[-1])

        # Ignore 0 weights
        benign = [x for x in weights if x[-1] < 0]
        malicious = [x for x in weights if x[-1] > 0][::-1]
        return weights, benign, malicious

    def perform_feature_selection(self, X_train, y_train):
        """Perform L2-penalty feature selection."""
        if self._num_features is not None:
            print('Performing L2-penalty feature selection')
            selector = LinearSVC(C=1)
            selector.fit(X_train, y_train)

            cols = np.argsort(np.abs(selector.coef_[0]))[::-1]
            cols = cols[:self._num_features]
        else:
            cols = [i for i in range(X_train.shape[1])]
        return cols

    def save_to_file(self):                
        with open(self.model_name, 'wb') as f:
            pickle.dump(self, f,protocol=4)
        if "svm-feature-selection" in self.model_name:
            model_name = os.path.join(cfg["models_training"],"svm-feature-selection-raw.p")        
            with open(model_name, 'wb') as f:
                pickle.dump(self.clf, f)
        elif "svm-feature-transformation" in self.model_name:
            model_name = os.path.join(cfg["models_training"],"svm-transformation-raw.p")        
            with open(model_name, 'wb') as f:
                pickle.dump(self.clf, f)


class SVM(SVMModel):
    """Standard linear SVM using scikit-learn implementation."""

    def __init__(self, malware_detector, selected_apps, num_features=None):
        super().__init__(malware_detector, selected_apps, num_features)
        self.model_name = self.generate_model_name()

    def fit(self, X_train, y_train):        
        #clf = LinearSVC(C=0.5)
        if "svm" not in self.malware_detector:
            clf = LinearSVC(C=1)          
            
            clf.fit(X_train, y_train)
            
        else:
           
            clf = LinearSVC(C=1)            
           
            clf.fit(X_train, y_train)
        return clf

    #Changed
    def generate_model_name(self):
        print("============================")
        print("self.malware_detector: ", self.malware_detector)
        print("============================")

        if self.malware_detector == "Drebin_feature_transformation":
            model_name = 'svm-feature-transformation'
        elif self.malware_detector == "Drebin_feature_selection":
            model_name = 'svm-feature-selection'
        elif "svm" in self.malware_detector:           
            model_name = 'svm'
        else:
            model_name = 'svm'
        model_name += '.p' if self._num_features is None else '-f{}.p'.format(self._num_features)
        
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print("model_name:",model_name)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        if self.malware_detector == "Drebin" or self.malware_detector == "Drebin_feature_transformation" or self.malware_detector == "Drebin_feature_transformation":
            return os.path.join(cfg['models_exp_robust_feature_representation'], model_name)
        else:
            return os.path.join(cfg['models_exp_adversarial_retraining'], model_name)

def load_from_file(model_filename):     
    with open(model_filename, 'rb') as f:
        return pickle.load(f)   
    
def vectorize(X, y):
    vec = DictVectorizer()# DictVectorizer(sparse=False)
    X = vec.fit_transform(X)
    y = np.asarray(y)
    return X, y, vec


    