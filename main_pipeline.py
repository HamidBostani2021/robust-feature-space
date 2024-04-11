# -*- coding: utf-8 -*-
"""
This tool is developed to improve the adversarial robustness of 
Android malware detection based on robust feature space.
"""
import numpy as np
import os
from timeit import default_timer as timer
import pandas as pd
from config import config as cfg
from tools import utils as utils_tools
import sys
import opf_construction as opf
import learner.models as models
from settings_new import config
import feature_transformation
import pickle
from sklearn.metrics import accuracy_score
from data_preparation import feature_set

def dict_to_feature_vector(d):
    vocab = utils_tools.read_pickle(cfg.get('feature.' + 'drebin', 'vocabulary'))
    features_List = [*d.keys()]
    vec_bool = [(feature in features_List) for feature in vocab]
    vec = list(map(int, vec_bool))
    vec_arr = np.asarray(vec)
    return vec_arr.reshape(1,len(vec))

def feature_vector_to_dict(x):
    vocab = utils_tools.read_pickle(cfg.get('feature.' + 'drebin', 'vocabulary'))    
    idx_features = [idx for idx,val in enumerate(x) if val == 1]
    d = dict()
    for idx in idx_features:
        d[vocab[idx]] = 1    
    return d

def f_importances(coef, names, top=-1):
    imp = coef
    original_features = names
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)  
    features = names[::-1][0:top]   
    feature_index = [idx for idx,val in enumerate(original_features) if val in features]
    feature_name = [val for idx,val in enumerate(original_features) if val in features]
    
    feature_name_index = list()#(range(n_top))
    
    for f in features:
        feature_name_index.append(feature_name.index(f))   
    return feature_index,feature_name

def load_dataset():
    trainX, valX, testX = utils_tools.read_joblib(cfg.get('feature.' + 'drebin', 'dataX'))
    trainy, valy, testy = utils_tools.read_joblib(cfg.get('feature.' + 'drebin', 'datay'))
    trainX = np.append(trainX,valX,axis=0)  
    trainy = np.append(trainy,valy,axis=0)  
    idx_malw = [idx for idx,val in enumerate(trainy) if val == 1]
    idx_beni = [idx for idx,val in enumerate(trainy) if val == 0]            
    X = np.append(trainX[idx_beni[0:45000],:],trainX[idx_malw[0:5000],:],axis = 0)        
    Y = np.append(trainy[idx_beni[0:45000]],trainy[idx_malw[0:5000]],axis = 0)
    idx = list(range(0,len(Y)))
    import random
    random.shuffle(idx)
    trainX = X[idx,:]
    trainy = Y[idx] 
    return trainX,trainy,testX,testy

def evaluating_robust_feature_representation(trainX,trainy,testX,testy,opt_list,opf_model,
                                             malware_app_indices,X,vocab,selected_apps,model_Drebin,meta,
                                             attack_name):
    if not (os.path.exists(cfg.get('feature.' + 'drebin', 'dataX_transformed'))):            
        print("train - transform")        
        trainX_transformed = feature_transformation.transform_sigmoid(trainX,opt_list,opf_model,1.0)
        print("------------")
        print("test - transform")             
        testX_transformed = feature_transformation.transform_sigmoid(testX,opt_list,opf_model,1.0)        
        print("------------")        
        utils_tools.dump_joblib([trainX_transformed, trainy, testX_transformed,testy],
                          cfg.get('feature.' + 'drebin', 'dataX_transformed'))
    else:
        path_x_trans = cfg.get('feature.' + 'drebin', 'dataX_transformed')        
        trainX_transformed, trainy,testX_transformed,testy = utils_tools.read_joblib(path_x_trans)
       
    
    model_Drebin_fs = models.SVM("Drebin_feature_selection", selected_apps,num_features = 500)   
    if os.path.exists(model_Drebin_fs.model_name):
        print("Load Drebin_feature_selection model ...")       
        model_Drebin_fs = models.load_from_file(model_Drebin_fs.model_name)
        trainX_fs = model_Drebin_fs.X_train
        testX_fs = model_Drebin_fs.X_test
        feature_index = model_Drebin_fs.column_idxs
        print("Execution Time (DREBIN-FeatureSelect) = %f"%model_Drebin_fs.execution_time)
        
        y_pred_app_new = model_Drebin_fs.clf.predict(testX_fs)    
        from sklearn.metrics import confusion_matrix
        tn_fs, fp_fs, fn_fs, tp_fs = confusion_matrix(testy,y_pred_app_new).ravel()
        tpr_fs = tp_fs/(tp_fs+fn_fs)
        fpr_fs = fp_fs/(fp_fs+tn_fs)           
      
       
    else:
        print("Generate Drebin_feature_selection model ...")
        trainX,trainy_fs,testX,testy_fs = load_dataset()
        feature_index = []
        from sklearn.svm import LinearSVC
        svm = LinearSVC(C=0.5)
        svm.fit(trainX, trainy_fs)
        n_top = 500#261
        feature_index,feature_name = f_importances(abs(svm.coef_[0]), vocab, top=n_top) 
        trainX_fs = trainX[:,feature_index]
        testX_fs = testX[:,feature_index]
        
        print("trainX_transformed.shape: ", trainX_fs.shape)
        print("testX_transformed.shape: ", testX_fs.shape)
        model_Drebin_fs.generate_transform(trainX_fs, testX_fs, trainy_fs, testy_fs,feature_index)   
        
        feature_index_path = os.path.join(config["models_training"],"svm-feature-selection-raw-feature_index.p")   
        with open(feature_index_path, 'wb') as f:
            pickle.dump(feature_index, f)
        
    y_pred_app_new = model_Drebin_fs.clf.predict(testX_fs)    
    from sklearn.metrics import confusion_matrix
    tn_fs, fp_fs, fn_fs, tp_fs = confusion_matrix(testy_fs,y_pred_app_new).ravel()
    tpr_fs = tp_fs/(tp_fs+fn_fs)
    fpr_fs = fp_fs/(fp_fs+tn_fs)
    print("TPR (DREBIN-FeatureSelect) = %f"%(tpr_fs*100))
    print("FPR (DREBIN-FeatureSelect) = %f"%(fpr_fs*100))
    print("Clean Acc (DREBIN-FeatureSelect) = %f"%(accuracy_score(testy_fs, y_pred_app_new)*100))
    print("------------------------------------------------------------------")
    
    
    model_name = os.path.join(config["models_training"],"secsvm-k0.2-lr0.0001-bs4096-e100-dataset_inaccessible-f10000-raw.p")  
    print("Load Sec-SVM model ...")
    with open(model_name, 'rb') as f:
        model_SecSVM_clf,execution_time = pickle.load(f)
    print("Execution Time (Sec-SVM) = %f"%execution_time)
    from scipy import sparse 
    test_set = sparse.csr_matrix(testX)
    y_pred_app = model_SecSVM_clf.predict(test_set)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(testy,y_pred_app).ravel()
    print('tp:', tp)
    print('tn:', tn)
    print('fn:', fn)
    print('fp:', fp)
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    print("TPR (DREBIN-SecSVM) = %f"%(tpr*100))
    print("FPR (DREBIN-SecSVM) = %f"%(fpr*100))
    print("Clean Acc (DREBIN-SecSVM) = %f"%(accuracy_score(testy, y_pred_app)*100))    
    print("------------------------------------------------------------------")
    
    model_Drebin_trans = models.SVM("Drebin_feature_transformation", selected_apps,num_features = len(opt_list.keys()))   
    if os.path.exists(model_Drebin_trans.model_name):
        print("Load Drebin_feature_transformation model ...")       
        model_Drebin_trans = models.load_from_file(model_Drebin_trans.model_name)      
        
        trainX_transformed = model_Drebin_trans.X_train
        testX_transformed = model_Drebin_trans.X_test
        print("Execution Time (Drebin_feature_transformation) = %f"%model_Drebin_trans.execution_time)       
    else:
        print("Generate Drebin_feature_transformation model ...")
        print("trainX_transformed.shape: ", trainX_transformed.shape)
        print("testX_transformed.shape: ", testX_transformed.shape)
        
        model_Drebin_trans.generate_transform(trainX_transformed, testX_transformed, trainy, testy,[])  
   
    
    y_pred_app_new = model_Drebin_trans.clf.predict(testX_transformed)    
    from sklearn.metrics import confusion_matrix
    tn_trans, fp_trans, fn_trans, tp_trans = confusion_matrix(testy,y_pred_app_new).ravel()
    tpr_trans = tp_trans/(tp_trans+fn_trans)
    fpr_trans = fp_trans/(fp_trans+tn_trans)
    print("TPR (DREBIN-Robust) = %f"%(tpr_trans*100))
    print("FPR (DREBIN-Robust) = %f"%(fpr_trans*100))
    print("Clean Acc (DREBIN-Robust) = %f"%(accuracy_score(testy, y_pred_app_new)*100))
    print("------------------------------------------------------------------")  
   
    
    y_pred = list()
    y_pred_secsvm = list()
    y_pred_trans = list()
    y_pred_fs = list()
    cnt = 0  
    no_f_drebin = dict()   
    no_f_secsvm = dict()
    no_f_drebin_trans = dict()
    no_f_drebin_fs = dict()
    for app_index in malware_app_indices:
        malware_dict = X[app_index]
        x_malware = dict_to_feature_vector(malware_dict) 
        y_pred_app = model_Drebin.clf.predict(x_malware)[0]   
        y_pred.append(y_pred_app)            
        no_f_drebin[meta[app_index]['pkg_name']+'.apk'] = sum(x_malware[0])
        
        y_pred_app_secsvm = model_SecSVM_clf.predict(sparse.csr_matrix(x_malware))[0]   
        y_pred_secsvm.append(y_pred_app_secsvm)            
        no_f_secsvm[meta[app_index]['pkg_name']+'.apk'] = sum(x_malware[0])        
        
        x_malware_trans = feature_transformation.transform_sigmoid(x_malware,opt_list,opf_model,1.0)
        y_pred_app_trans = model_Drebin_trans.clf.predict(x_malware_trans)[0]
        y_pred_trans.append(y_pred_app_trans)            
        no_f_drebin_trans[meta[app_index]['pkg_name']+'.apk'] = sum(x_malware_trans[0])    
        
        x_malware_fs = x_malware[:,feature_index]        
        y_pred_app_fs = model_Drebin_fs.clf.predict(x_malware_fs)[0]   
        y_pred_fs.append(y_pred_app_fs)                
        no_f_drebin_fs[meta[app_index]['pkg_name']+'.apk'] = sum(x_malware_fs[0])
        
        cnt += 1
        if cnt % 100 ==0:
            print("cnt = ",cnt)
    DR = (sum(y_pred)/len(malware_app_indices))*100   
    DR_secsvm = (sum(y_pred_secsvm)/len(malware_app_indices))*100
    DR_trans = (sum(y_pred_trans)/len(malware_app_indices))*100
    DR_fs = (sum(y_pred_fs)/len(malware_app_indices))*100
    print("DR DREBIN-Original:" + str(DR) + "%")
    print("DR Sec-SVM:" + str(DR_secsvm) + "%")
    print("DR DREBIN-Robust:" + str(DR_trans) + "%")    
    print("DR DREBIN-FeatureSelect:" + str(DR_fs) + "%")    
    
    mal_detected_idx = [idx for idx,val in enumerate(y_pred) if val == 1]
    mal_detected_idx_trans = [idx for idx,val in enumerate(y_pred_trans) if val == 1]        
    
    mal_detected_idx_secsvm = [idx for idx,val in enumerate(y_pred_secsvm) if val == 1]        
    mal_detected_idx_fs = [idx for idx,val in enumerate(y_pred_fs) if val == 1]
    mal_detected_idx_common = [val for val in mal_detected_idx if val in mal_detected_idx_trans and val in mal_detected_idx_fs and val in mal_detected_idx_secsvm]    
    
    #mal_detected_idx_common = [val for val in mal_detected_idx if val in mal_detected_idx_trans]    
    
    malware_app_indices_detected = [val for idx,val in enumerate(malware_app_indices) if idx in mal_detected_idx_common]
    mal_detected_name_common = [val['pkg_name'] for idx,val in enumerate(meta) if idx in malware_app_indices_detected]
    print("common detected apps = ",len(mal_detected_idx_common))
    
    apps_adv_malware_path = os.path.join(config["adversarial_examples"],attack_name, "apps_adv_malware.pkl")
    with open(apps_adv_malware_path, 'rb') as f:
        apps_adv_malware = pickle.load(f)
    
    apps_malware_path = os.path.join(config["adversarial_examples"],attack_name,"apps_malware.pkl")
    with open(apps_malware_path, 'rb') as f:
        apps_malware = pickle.load(f)
    
    y_pred_adv = list()
    y_pred_adv_secsvm = list()
    y_pred_adv_trans = list()
    y_pred_adv_fs = list()
    cnt = 0   
    total_added_no_f_drebin = 0 
    total_added_no_f_secsvm = 0 
    total_added_no_f_drebin_trans = 0
    total_added_no_f_drebin_fs = 0
    adv_no = 0    
    
    for key in apps_adv_malware.keys():
        if key not in mal_detected_name_common:                
           continue       
       
        adv_dict = apps_adv_malware[key]
        x_malware_adv = dict_to_feature_vector(adv_dict)       
        y_pred_app = model_Drebin.clf.predict(x_malware_adv)[0]  
        y_pred_adv.append(y_pred_app)
        adv_no = len([val for val in y_pred_adv if val == 0])        
        if y_pred_app == 0:             
            if len(apps_adv_malware[key]) > len(apps_malware[key]):                
                malware_dict = apps_malware[key]
                x_malware = dict_to_feature_vector(malware_dict)[0]               
                total_added_no_f_drebin += (sum(x_malware_adv[0])-sum(x_malware))
        
        
        y_pred_app_secsvm = model_SecSVM_clf.predict(sparse.csr_matrix(x_malware_adv))[0]  
        y_pred_adv_secsvm.append(y_pred_app_secsvm)
        adv_no_secsvm = len([val for val in y_pred_adv_secsvm if val == 0])        
        if y_pred_app_secsvm == 0:         
            if len(apps_adv_malware[key]) > len(apps_malware[key]):                
                malware_dict = apps_malware[key]
                x_malware = dict_to_feature_vector(malware_dict)[0]
                total_added_no_f_secsvm += (sum(x_malware_adv[0])-sum(x_malware))
        
        x_malware__adv_trans = feature_transformation.transform_sigmoid(x_malware_adv,opt_list,opf_model,1.0)
        y_pred_app_trans = model_Drebin_trans.clf.predict(x_malware__adv_trans)[0]   
        y_pred_adv_trans.append(y_pred_app_trans)
        adv_no_trans = len([val for val in y_pred_adv_trans if val == 0])       
        if y_pred_app_trans == 0:            
            if len(apps_adv_malware[key]) > len(apps_malware[key]):               
                malware_dict = apps_malware[key]
                x_malware = dict_to_feature_vector(malware_dict)[0]
                total_added_no_f_drebin_trans += (sum(x_malware_adv[0])-sum(x_malware))
        
        
        x_malware__adv_fs = x_malware_adv[:,feature_index]
        y_pred_app_fs = model_Drebin_fs.clf.predict(x_malware__adv_fs)[0]   
        y_pred_adv_fs.append(y_pred_app_fs)
        adv_no_fs = len([val for val in y_pred_adv_fs if val == 0])       
        if y_pred_app_fs == 0:            
            if len(apps_adv_malware[key]) > len(apps_malware[key]):
                malware_dict = apps_malware[key]
                x_malware = dict_to_feature_vector(malware_dict)[0]
                total_added_no_f_drebin_fs += (sum(x_malware_adv[0])-sum(x_malware))
        
        
        cnt += 1
    print("cnt = ",cnt) 
    
    ER_DREBIN_Original = (adv_no/len(y_pred_adv))*100
    ER_DREBIN_FeatureSelect = (adv_no_fs/len(y_pred_adv_fs))*100
    ER_Sec_SVM = (adv_no_secsvm/len(y_pred_adv_secsvm))*100
    ER_DREBIN_Robust = (adv_no_trans/len(y_pred_adv_trans))*100
    
    print("Roubust Acc DREBIN-Original = %f"%(100 - ER_DREBIN_Original))    
    print("Roubust Acc DREBIN-FeatureSelect = %f"%(100-ER_DREBIN_FeatureSelect))
    print("Roubust Acc Sec-SVM = %f"%(100-ER_Sec_SVM))
    print("Roubust Acc DREBIN-Robust = %f"%(100-ER_DREBIN_Robust))
    print("-------------------------------")
    print("Avg. Added Features - DREBIN-Original = %f"%(total_added_no_f_drebin/adv_no))    
    print("Avg. Added Features - DREBIN-FeatureSelect = %f"%(total_added_no_f_drebin_fs/adv_no_fs))
    print("Avg. Added Features - Sec-SVM = %f"%(total_added_no_f_secsvm/adv_no_secsvm))
    print("Avg. Added Features - DREBIN-Robust= %f"%(total_added_no_f_drebin_trans/adv_no_trans)) 
    
def measure_phi_coefficient(no_features):
    trainX, valX, testX = utils_tools.read_joblib(cfg.get('feature.' + 'drebin', 'dataX'))
    trainy, valy, testy = utils_tools.read_joblib(cfg.get('feature.' + 'drebin', 'datay'))
    data = np.append(trainX,valX,axis=0)  
    
    data = data[:,0:no_features]
    df = pd.DataFrame(data)               
    #corr_matrix_temp = df.corr(method = 'spearman') 
    corr_matrix_path = '/home/hamid/Projects/Android_AML_Robustness/data/drebin/corr_matrix.pkl'
    #corr_matrix_path = 'C:/GitHub/Android_AML_Robustness/data/drebin/corr_matrix.pkl'
    col = 0
    if os.path.exists(corr_matrix_path) == True:
        print("corr_matrix_path exists...")
        corr_matrix = utils_tools.read_joblib(corr_matrix_path)
        col = corr_matrix.shape[0]                
    else:
        corr_matrix = np.zeros((no_features,no_features))
    print("col = ",col)
    for i in range(col,no_features):
        start = timer()
        feature = list(range(i,no_features))
        data_temp = data[:,feature]
        df = pd.DataFrame(data_temp)    
        if i == 0:
            corr_matrix = np.array(df.corrwith(df[i],method = 'spearman'))
        else:
            #corr_matrix_temp = np.array(df.corrwith(df[i],method = 'spearman'))
            prefix = np.zeros((1,i))
            corr_matrix_temp = np.array(df.corrwith(df[0],method = 'spearman'))
            corr_matrix_temp = np.append(prefix,[corr_matrix_temp],axis = 1)[0]            
            corr_matrix = np.vstack((corr_matrix,corr_matrix_temp))        
        if i % 10 == 0:
            corr_matrix_path = os.path.join(cfg.get('feature.' + 'drebin', 'corr_matrix_path'),'corr_matrix.pkl')
            utils_tools.dump_joblib(corr_matrix,corr_matrix_path)
        end = timer()
        execution_time = end - start
        print("execution_time = ", execution_time)
        print("i = %d"%(i))
        print("----------")
        
    corr_matrix_path = os.path.join(cfg.get('feature.' + 'drebin', 'corr_matrix_path'),'corr_matrix.pkl')
    utils_tools.dump_joblib(corr_matrix,corr_matrix_path)   
    print("Finish")
    return corr_matrix

if __name__ =="__main__":
    attack_name = sys.argv[1]
    adv_examples_types = sys.argv[2]
    correlation_threshold = sys.argv[3]
    adv_examples_types = "PierazziAttack"#"PierazziAttack"#EvadeDroid
    attack_name = "PierazziAttack"#"EvadeDroid"#"PierazziAttack"
    feature_dict_name = "apps_features_for_retraining_exp.pkl"#"apps_features_for_fs_exp.pkl"#"apps_features_for_trans_exp.pkl"       
    data_preparation = False
    measure_correlation = False
    exp_robust_feature_representation = True    
    exp_name = "exp_data_augmentation"#"exp_data_augmentation"#"exp_robust_feature_representation"#"exp_data_augmentation_pairwise_clustering"            
    
    if data_preparation == True:
        feature_set.determine_smaples_training_test(5000, 25000)    
        feature_set._data_preprocess()
    corr_matrix_path = os.path.join(cfg.get('feature.' + 'drebin', 'corr_matrix_path'),'corr_matrix.pkl')
    if os.path.exists(corr_matrix_path) == True:
        corr_matrix = measure_phi_coefficient(10000)
    else:
        corr_matrix = utils_tools.read_joblib(corr_matrix_path)
    prototypes_path = os.path.join(cfg.get('feature.' + 'drebin', 'corr_matrix_path'),'prototypes.pkl')
    if os.path.exists(prototypes_path) == True:
        prototypes = utils_tools.read_joblib(prototypes_path)
    else:
        prototypes = opf.find_prototypes(corr_matrix)    
        utils_tools.dump_joblib(prototypes,prototypes_path)
        
    corr_matrix_full = corr_matrix + np.transpose(corr_matrix) + np.diag(np.full(corr_matrix.shape[0],1))
    
    opf_model_path = os.path.join(cfg.get('feature.' + 'drebin', 'corr_matrix_path'),'opf_model.pkl')
    if os.path.exists(opf_model_path) == True:
        opf_model = utils_tools.read_joblib(opf_model_path)
    else:        
        opf_model = opf.train(corr_matrix_full,prototypes)
        utils_tools.dump_joblib(opf_model,opf_model_path)    
   
    cost_threshold = 100
    opt_list = dict()
    for i in opf_model[:,0]:
        i = round(i)
        if round(opf_model[i,4]) not in [*opt_list.keys()]:
            if opf_model[i,1] <= cost_threshold:
                opt_list[round(opf_model[i,4])] = [round(opf_model[i,0])]
        else:
            if opf_model[i,1] <= cost_threshold:
                opt_list[round(opf_model[i,4])].append(round(opf_model[i,0]))
    vocab = utils_tools.read_pickle(cfg.get('feature.' + 'drebin', 'vocabulary'))
    trainX, valX, testX = utils_tools.read_joblib(cfg.get('feature.' + 'drebin', 'dataX'))
    trainy, valy, testy = utils_tools.read_joblib(cfg.get('feature.' + 'drebin', 'datay'))
    trainX = np.append(trainX,valX,axis=0)  
    trainy = np.append(trainy,valy,axis=0)  
    selected_apps = list(range(0,len(trainy)))   
    
   
    print("------------------------------------------------------------------")
    model_Drebin = models.SVM("Drebin", selected_apps,num_features = 10000)    
    features_name = vocab
    features_index = list(range(0,10000))
    if os.path.exists(model_Drebin.model_name):
        print("Load DREBIN-Original model ...")
        model_Drebin = models.load_from_file(model_Drebin.model_name)
        no_benign = len([val for val in model_Drebin.y_train if val == 0])
        no_malware = len([val for val in model_Drebin.y_train if val == 1])
        print("no_benign = ", no_benign)
        print("no_malware = ", no_malware)            
        trainX = model_Drebin.X_train
        trainy = model_Drebin.y_train
        idx = [idx for idx,val in enumerate(trainy) if val == 1]
        print("Drebin",sum(sum(trainX[idx,:])))   
        print("Execution Time (DREBIN-Original) = %f"%model_Drebin.execution_time)
    else:           
        idx_malw = [idx for idx,val in enumerate(trainy) if val == 1]
        idx_beni = [idx for idx,val in enumerate(trainy) if val == 0]            
        X = np.append(trainX[idx_beni[0:45000],:],trainX[idx_malw[0:5000],:],axis = 0)        
        Y = np.append(trainy[idx_beni[0:45000]],trainy[idx_malw[0:5000]],axis = 0)
        idx = list(range(0,len(Y)))
        import random
        random.shuffle(idx)
        trainX = X[idx,:]
        trainy = Y[idx]          
        print("Generate DREBIN-Original model ...")
        model_Drebin.generate(trainX, testX, trainy, testy,features_name,features_index)           
   
    y_pred_app = model_Drebin.clf.predict(testX)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(testy,y_pred_app).ravel()
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    print("TPR (DREBIN-Original) = %f"%(tpr*100))
    print("FPR (DREBIN-Original) = %f"%(fpr*100))
    print("Clean Acc (DREBIN-Original) = %f"%(accuracy_score(testy, y_pred_app)*100))
    print("------------------------------------------------------------------")        
   
    
    import ujson as json       
    X_filename = config["X_dataset_accessible"]
    with open(X_filename,'rb') as f:
        X = json.load(f)        
    Y_finename = config["Y_dataset_accessible"]
    with open(Y_finename,'rt') as f:
        Y = json.load(f)
    meta_filename = config["meta_accessible"]
    with open(meta_filename, 'rt') as f:
        meta = json.load(f)
        
    malware_app_indices = [idx for idx,val in enumerate(Y) if val == 1] 
    evaluating_robust_feature_representation(trainX,trainy,testX,testy,opt_list,
                                             opf_model,malware_app_indices,X,vocab,selected_apps,
                                             model_Drebin,meta,attack_name)  
   
  