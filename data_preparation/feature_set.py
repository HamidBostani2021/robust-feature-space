# -*- coding: utf-8 -*-

"""
Data preparation
"""
import config as conf
from data_preparation.feature_extractor import FeatureMapping
from sklearn.model_selection import train_test_split
from tools import utils
import sys
import numpy as np
import os
from settings_new import config as cfg
import json
import random

#DREBIN20 dataset
#No.Total = 151,637
#No. clean samples = 135,859
#No. malware samples = 15,778


def determine_smaples_training_test(no_malware_test, no_clean_test):
    
    path = os.path.join(cfg['features_training'],'training-dataset-X.json')
    if os.path.exists(path):
        return
    
    X_filename = os.path.join(cfg['total_dataset'] , 'apg-X.json')   
    with open(X_filename, 'rb') as f:
        X = json.load(f)
        
    y_filename = os.path.join(cfg['total_dataset'], 'apg-Y.json')   
    with open(y_filename, 'rt') as f:
        Y = json.load(f)
    
    meta_filename = os.path.join(cfg['total_dataset'], 'apg-meta.json')   
    with open(meta_filename, 'rt') as f:
        meta = json.load(f)   
    
    
    
    goodware_index = [idx for idx,val in enumerate(Y) if val == 0]
    malware_index = [idx for idx,val in enumerate(Y) if val == 1]
    
    selected_goodware_app_test_index = random.sample(range(0,len(goodware_index)),no_clean_test)
    selected_goodware_app_test = [goodware_index[idx] for idx in selected_goodware_app_test_index]
    
    selected_malware_app_test_index = random.sample(range(0,len(malware_index)),no_malware_test)
    selected_malware_app_test = [malware_index[idx] for idx in selected_malware_app_test_index]
    
    test_apps_index = selected_goodware_app_test + selected_malware_app_test

    X_test = [x for idx,x in enumerate(X) if idx in test_apps_index]
    path = os.path.join(cfg['features_test'],'test-dataset-X.json')
    with open(path, 'w') as f:
        json.dump(X_test,f)
    
    Y_test = [y for idx,y in enumerate(Y) if idx in test_apps_index]
    path = os.path.join(cfg['features_test'],'test-dataset-Y.json')
    with open(path, 'w') as f:
        json.dump(Y_test,f)
    
    meta_test = [m for idx,m in enumerate(meta) if idx in test_apps_index]
    path = os.path.join(cfg['features_test'],'test-dataset-meta.json')
    with open(path, 'w') as f:
        json.dump(meta_test,f)
        
    
    
    selected_goodware_app_training_index = list(range(0,len(goodware_index)))
    selected_goodware_app_training = [goodware_index[idx] for idx in selected_goodware_app_training_index if idx not in selected_goodware_app_test_index]
    
    selected_malware_app_training_index = list(range(0,len(malware_index)))
    selected_malware_app_training = [malware_index[idx] for idx in selected_malware_app_training_index if idx not in selected_malware_app_test_index]
    
            
    training_apps_index = selected_goodware_app_training + selected_malware_app_training  
        
    X_training = [x for idx,x in enumerate(X) if idx in training_apps_index]
    path = os.path.join(cfg['features_training'],'training-dataset-X.json')
    with open(path, 'w') as f:
        json.dump(X_training,f)
    
    Y_training = [y for idx,y in enumerate(Y) if idx in training_apps_index]
    path = os.path.join(cfg['features_training'],'training-dataset-Y.json')
    with open(path, 'w') as f:
        json.dump(Y_training,f)
    
    meta_training = [m for idx,m in enumerate(meta) if idx in training_apps_index]
    path = os.path.join(cfg['features_training'],'training-dataset-meta.json')
    with open(path, 'w') as f:
        json.dump(meta_training,f)
        

def _compute_no_features():
             
        path = cfg['features_training']             
        X_filename = os.path.join(path,"training-dataset-X.json")
        with open(X_filename,'rb') as f:
            X = json.load(f)
        
        Y_finename = os.path.join(path,"training-dataset-Y.json")
        with open(Y_finename,'rt') as f:
            Y = json.load(f)       
        
       
        _, Y, vec = vectorize(X, Y) 
        vocab = vec.feature_names_       
        print(f"no. features: {len(vocab)}")

def _data_preprocess():
        """
        feature extraction
        """
        feature_tp = 'drebin'
        path = conf.get('feature.' + feature_tp, 'dataX')
        if os.path.exists(path) == True:
            return
        try:                 
            path = cfg['features_training']             
            X_filename = os.path.join(path,"training-dataset-X.json")
            with open(X_filename,'rb') as f:
                X = json.load(f)
            
            Y_finename = os.path.join(path,"training-dataset-Y.json")
            with open(Y_finename,'rt') as f:
                Y = json.load(f)
            
            
            meta_filename = os.path.join(path,"training-dataset-meta.json")
            with open(meta_filename, 'rt') as f:
                meta = json.load(f)
            
            _, Y, vec = vectorize(X, Y) 
            vocab = vec.feature_names_
            vocab_info_dict = dict([(val,val.split("::")[0]) for val in vocab])
            name_list = [val["sha1"] for val in meta]
            features = X#.toarray()
            gt_label = Y                      
           
            path = cfg['features_test']             
            X_filename = os.path.join(path,"test-dataset-X.json")
            with open(X_filename,'rb') as f:
                X = json.load(f)
            
            Y_finename = os.path.join(path,"test-dataset-Y.json")
            with open(Y_finename,'rt') as f:
                Y = json.load(f)
            
            
            meta_filename = os.path.join(path,"test-dataset-meta.json")
            with open(meta_filename, 'rt') as f:
                meta = json.load(f)
            
            _, Y, vec = vectorize(X, Y)             
            name_list_test = [val["sha1"] for val in meta]
            features_test = X#.toarray()
            gt_label_test = Y
            
            print("data loading is completed ...")
            
            #Hamid
            # select frequent features
            data_root_dir = conf.get("dataset", "dataset_root")
            feat_save_dir = os.path.join(data_root_dir, "apk_data")
            feature_mapping = FeatureMapping(feat_save_dir, feature_type="drebin")      
           
            vocab_selected, vocab_info_dict_selcted = \
                 feature_mapping.select_feature(features, gt_label, vocab, vocab_info_dict, dim=10000)            
          
            print("feature selection-part 1 is completed ...")
             
            features = \
                    feature_mapping.binary_feature_mapping_normalized(vocab_selected, features, status='train')    
              
            features_test = \
                    feature_mapping.binary_feature_mapping_normalized(vocab_selected, features_test, status='train')    

            
            print("feature selection is completed ...") 
            
            print("start feature splitting 1 ................")
           
            train_features=features
            train_y=gt_label
            train_name_list=name_list
            test_features = features_test
            test_y = gt_label_test
            test_name_list = name_list_test
            
            print("start feature splitting 2 ................")
            train_features, val_features, train_y, val_y, train_name_list, val_name_list = \
                train_test_split(train_features, train_y, train_name_list, test_size=0.25, random_state=0)
            
            print("feature splitting is completed ...")        
           
            training_feature_vectors=train_features
            val_feature_vectors=val_features
            test_feature_vectors=test_features
            
            # save features and feature representations            
            utils.dump_pickle(vocab_selected, conf.get('feature.' + feature_tp, 'vocabulary'))
            utils.dump_pickle(vocab_info_dict_selcted, conf.get('feature.' + feature_tp, 'vocab_info'))
            utils.dump_joblib([training_feature_vectors, val_feature_vectors, test_feature_vectors],
                              conf.get('feature.' + feature_tp, 'dataX'))
            utils.dump_joblib([train_y, val_y, test_y],
                              conf.get('feature.' + feature_tp, 'datay'))

            utils.write_whole_file('\n'.join(train_name_list + val_name_list + test_name_list),
                                  conf.get('dataset', 'name_list'))
            
            print("save features is completed ...")
           
        except Exception as ex:
            print(ex)          
            sys.exit(1)


def vectorize(X, y):
    from sklearn.feature_extraction import DictVectorizer
    vec = DictVectorizer()# DictVectorizer(sparse=False)
    X = vec.fit_transform(X)
    y = np.asarray(y)
    return X, y, vec




