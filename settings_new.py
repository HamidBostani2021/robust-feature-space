# -*- coding: utf-8 -*-

"""
Configuring pipeline.
"""
import os

_absolute_project_path = 'C:/GitHub/Android-adversarial-robustness-with-domain-constraints-main-draft-for-ccs2023'
#_absolute_project_path = 'Projects/Android-adversarial-robustness-with-domain-constraints'
_absolute_java_components_path = 'EvadeDroid/java-components/build'
_absolute_graph_path = '/Result_Graphs'

def project(base):
    return os.path.join(_absolute_project_path, base)


def java_components(base):
    return os.path.join(_absolute_java_components_path, base)


config = {    
    'project_root': _absolute_project_path,
    'result_graphs': _absolute_graph_path,
    
    # data: apks and features
    'adversarial_examples': project('data/adversarial_examples/'),
    'apks_test': project('data/apks/test/'),
    'apks_training': project('data/apks/training/'),    
    'features_test': project('data/features/test/'),
    'features_training': project('data/features/training/'),   
    'features' : project('data/features/'),  
    'apks': project('data/apks/'),  
    'stored_components': project('data/stored-components/'),    
    'mamadroid':project('mamadroid'),      
    'models_test': project('data/models/test/'),
    'models_training': project('data/models/exp_robust_feature_representation/'),
    'models_training_augment': project('data/models/exp_adversarial_retraining/'),
    'total_dataset': project('data/features/total/'),    
    'X_dataset_test': project('data/features/test/test-dataset-X.json'),
    'Y_dataset_test': project('data/features/test/test-dataset-Y.json'),
    'meta_test': project('data/features/test/test-dataset-meta.json'),    
    'X_dataset_training': project('data/features/training/training-dataset-X.json'),
    'Y_dataset_training': project('data/features/training/training-dataset-Y.json'),
    'meta_training': project('data/features/training/training-dataset-meta.json'), 
    
    'X_dataset_accessible': project('data/features/accessible/accessible-dataset-X.json'),
    'Y_dataset_accessible': project('data/features/accessible/accessible-dataset-Y.json'),
    'meta_accessible': project('data/features/accessible/accessible-dataset-meta.json'),     
    
    #'feature_extractor': project('drebin_feature_extractor'),
    'feature_extractor': '/home/hamid/Projects/feature-extractor',
    'tmp_dir': project('data/stored-components/tmp/'),  
    'goodware_location': project('/data/apk'),

    # Other components: software transplantation and mamadroid
    'soot':java_components('soot/'),   
    'extractor': java_components('extractor.jar'),
    'appgraph': java_components('appgraph.jar'),
    'ice_box': project('/data/stored-components/ice_box/'),
    'android_sdk': '/usr/lib/android-sdk/',
    'java_sdk': '/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/',    
    'extractor_timeout': 600,
    'template_path': project('data/template'),
    'results_dir': project('/data/stored-components/attack-results'),   
    'indices': project(''),  # only needed if using fixed indices   
    'injector': java_components('injector.jar'),
    'smallinjector': java_components('smallinjector.jar'),
    'template_injector': java_components('templateinjector.jar'),
    'cc_calculator': java_components('cccalculator.jar'),
    'class_lister': java_components('classlister.jar'),
    'classes_file': project('all_classes.txt'),       
    'mined_slices': project('data/stored-components/mined-slices'),
    'opaque_pred': project('opaque-preds/sootOutput'),
    'resigner': java_components('apk-signer.jar'),      
    'cc_calculator_timeout': 600,  
    'storage_radix': 0,  # Use if apps are stored with a radix (e.g., radix 3: root/0/0/A/00A384545.apk)
    
    'models_exp_robust_feature_representation': project('data/models/exp_robust_feature_representation/'),
    'models_exp_adversarial_retraining': project('data/models/exp_adversarial_retraining/'),
    'models_inaccessible': project('data/models/inaccessible/'),
    
    
    
    # Miscellaneous options
    'tries': 1,
    'nprocs_preload': 8,
    'nprocs_evasion': 12,
    'nprocs_transplant': 8
}
