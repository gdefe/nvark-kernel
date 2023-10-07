import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit




# SVM

def my_SVM_classifier(Ktrtr, Ktetr, TRAIN_y, TEST_y, svm_C, verbose=True):
    """ perform SVM classification with precomputed kernel
    
    The advantages of support vector machines are:
    -- Effective in high dimensional spaces.
    -- Still effective in cases where number of dimensions is greater than the number of samples.
    -- Uses a subset of training points in the decision function (called support vectors), so it is 
        also memory efficient.
    -- Versatile: different Kernel functions can be specified for the decision function.

    The disadvantages of support vector machines include:
    -- If the number of features is much greater than the number of samples, avoid over-fitting 
        in choosing Kernel functions and regularization term is crucial.
    -- SVMs do not directly provide probability estimates, these can be calculated using an expensive
        five-fold cross-validation.
    
    Parameters:
        Ktrtr (N_train x N_train): train-train kernel matrix
        Ktetr (N_test x N_train):  test-train kernel matrix
        TRAIN_y (N_test x N_labels):  train labels
        TEST_y (N_test x N_labels):   test labels
    
    Returns:
        accuracy (float): SVM classification accuracy
    """
    # In SVC, if the data is unbalanced (e.g. many positive and few negative), set class_weight='balanced' and/or try different penalty parameters C
    # Set the parameter C of class i to class_weight[i]*C for SVC
    # The “balanced” mode uses the values of y to automatically adjust weights 
    # inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
    # svm = SVC(C=svm_C, kernel='precomputed', class_weight='balanced', cache_size = 500) 
    
    # one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as
    # all other classifiers, or the original one-vs-one (‘ovo’) decision function of
    # libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).
    # svm = SVC(C=svm_C, kernel='precomputed', class_weight=None, decision_function_shape='ovo', cache_size = 500) 
    
    svm = SVC(C=svm_C, kernel='precomputed', class_weight=None, decision_function_shape='ovr', cache_size = 500) 
    
    svm.fit(Ktrtr, TRAIN_y)
    labels_pred = svm.predict(Ktetr)
    
    accuracy = np.sum(labels_pred == TEST_y)/len(labels_pred)
    if verbose: print('SVM Accuracy = %.3f'%(accuracy))
    return accuracy


def my_SVMopt_classifier(Ktrtr, TRAIN_y, 
                         Ktetr=None, TEST_y=[], 
                         svm_C_list=[5.0], 
                         random_state=1234, n_folds=10, val_size=0.3, 
                         verbose=False):
    """ perform SVM classification with precomputed kernel, optimizing some parameters
    
    Parameters:
        Ktrtr   (N_train x N_train):  train-train kernel matrix
        Ktetr   (N_test x N_train):   test-train kernel matrix
        TRAIN_y (N_test x N_labels):  train labels
        TEST_y  (N_test x N_labels):  test labels
        svm_C_list     (list):        list of alloweed values
                
    Returns:
        acc_test       (float): SVM classification accuracy on test with best values
        acc_train      (float): SVM classification accuracy on train with best values
        svm_C_best     (float): best C
        acc_over_pars  (len(svm_C_list) x 3): SVM classification accuracy on train over parameter settings
    """  
    acc_matrix = np.zeros((1,len(svm_C_list)))
    acc_over_pars = np.zeros((len(svm_C_list),2))
    
    # split the training data into stratified randomized folds
    sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=val_size, random_state=random_state)
    
    # split the training data into stratified randomized folds
    # sss = StratifiedKFold(n_splits=n_splits)
    
    for i, C in enumerate(svm_C_list):
        acc_over_folds = [] 
        for cv_train_i, cv_test_i in sss.split(np.arange(Ktrtr.shape[0]), TRAIN_y):
            cv_train_i = np.sort(cv_train_i)
            cv_test_i  = np.sort(cv_test_i)
            # pick Ktrtr and split into train 
            Ktrtr_fold = Ktrtr[cv_train_i,:]
            Ktrtr_fold = Ktrtr_fold[:,cv_train_i]
            TRAIN_y_fold = TRAIN_y[cv_train_i]
            # and test
            Ktetr_fold = Ktrtr[cv_test_i,:]
            Ktetr_fold = Ktetr_fold[:,cv_train_i]
            TEST_y_fold = TRAIN_y[cv_test_i]

            acc_over_folds.append(my_SVM_classifier(Ktrtr_fold, Ktetr_fold, TRAIN_y_fold, TEST_y_fold, C, verbose=False))
        mean = np.mean(acc_over_folds)
        acc_matrix[0,i] = mean
        acc_over_pars[i,:] = [C,mean]
    
    best_i = np.argmax(acc_over_pars[:,1])
    best_C   = acc_over_pars[best_i,0]
    acc_train = acc_over_pars[best_i,1]
    if verbose: print('SVM Accuracy on Train = %.3f'%(acc_train))
    
    # calculate accuracy on test if given
    acc_test = None
    if Ktetr is not None and TEST_y is not []:
        acc_test = my_SVM_classifier(Ktrtr, Ktetr, TRAIN_y, TEST_y, best_C, verbose=verbose)
    return acc_test, acc_train, best_C