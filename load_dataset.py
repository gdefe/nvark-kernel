import numpy as np
import pandas as pd

# Load MTS datasets
from sktime.datasets import load_from_tsfile
from sktime.datasets import load_from_ucr_tsv_to_dataframe 

# interpolate
from sktime.transformations.panel.interpolate import TSInterpolator

# One hot encoder
from sklearn.preprocessing import OneHotEncoder


def load_dataset(dataset_name):
        # load from archives
        success=False
        try:
            TRAIN_x_raw, TRAIN_y_raw = load_from_ucr_tsv_to_dataframe("datasets/UCR 2018 Archive/"+dataset_name+"/"+dataset_name+"_TRAIN.tsv")
            TEST_x_raw,  TEST_y_raw  = load_from_ucr_tsv_to_dataframe("datasets/UCR 2018 Archive/"+dataset_name+"/"+dataset_name+"_TEST.tsv")      
            success=True
        except: pass

        try:
            TRAIN_x_raw, TRAIN_y_raw = load_from_tsfile("datasets/UEA 2018 Archive/"+dataset_name+"/"+dataset_name+"_TRAIN.ts")
            TEST_x_raw,  TEST_y_raw  = load_from_tsfile("datasets/UEA 2018 Archive/"+dataset_name+"/"+dataset_name+"_TEST.ts") 
            success=True
        except: pass
        
        try:
            TRAIN_x_raw, TRAIN_y_raw = load_from_tsfile("datasets/UCR 2018 Archive/"+dataset_name+"/"+dataset_name+"_TRAIN.ts")
            TEST_x_raw,  TEST_y_raw  = load_from_tsfile("datasets/UCR 2018 Archive/"+dataset_name+"/"+dataset_name+"_TEST.ts")
            success=True
        except: pass
    
        if not success:
            raise RuntimeError(f'Could not load dataset "{dataset_name}"')
        else:
            return TRAIN_x_raw, TRAIN_y_raw, TEST_x_raw, TEST_y_raw  
        
        
  
    
        
def dataset_has_nans(_dataset):
    dataset = _dataset.copy()
    dataset = dataset.applymap( lambda series: (np.array(np.isnan(series))==True).any() )
    return dataset.values.any()
    
def different_lengths(_dataset):
    dataset = _dataset.copy()
    dataset = dataset.applymap(lambda series: len(series))
    first_value = dataset.iloc[0,0]
    flag = (dataset==first_value).values.all()
    T_min = dataset.values.min() 
    T_max = dataset.values.max()
    return flag, T_min, T_max
        
def print_info(dataset_name, TRAIN, TEST, y=None):    
    # T can vary across N; D must be the same across N
    info = {}          # contains the (NxTxD) shape infos and if different length
    print('{:<37}  {:<8}{:<14}  {:<8}{:<8}'.format(f'\n### {dataset_name} ###', 'N', 'T_min|T_max', 'D', 'N_class'))
    print('------------------------------------------------------------------------------------------')
    traintest = {f'{dataset_name} train':TRAIN, f'{dataset_name} test':TEST}
    for key, dataset in traintest.items():
        info[key] = {}
        info[key]['N'] = len(dataset.index)
        info[key]['equalL'], info[key]['T_min'], info[key]['T_max'] = different_lengths(dataset)   
        info[key]['D'] = len(dataset.columns)                
        info[key]['nans'] = dataset_has_nans(dataset)     
        
        line_info = '{:<37}  {:<8} {:<14} {:<8} {:<8} {:<8}'.format(key, info[key]['N'], \
                                                                    str(info[key]['T_min'])+'|'+str(info[key]['T_max']), \
                                                                        info[key]['D'], len(set(y)), \
                                                                            " nans = "+str(info[key]['nans']))
        print(line_info)
    return info







"""############### Preprocessing ########################################################################################"""
####### Normalzie ########################################################################### 
# for each dataset, we subtract the mean and divide by the standard deviation within each dimension
def normalize_MTS(_train, _test):
    train = _train.copy()
    test  = _test.copy()
    N_train = train.shape[0]
    D       = train.shape[1]
    for d in range(D):
        train_red = train.iloc[:,d]
        test_red  = test.iloc[:,d]
        long_series = pd.Series([], dtype=object)
        for i in range(N_train):
            long_series = pd.concat((long_series, pd.Series(train_red.iloc[i])), ignore_index=True)
        long_series = long_series.to_numpy()
        mean, std = np.nanmean(long_series), np.nanstd(long_series)
        # replace
        train.iloc[:,d] = (train_red - mean)/std
        test.iloc[:,d] =  (test_red - mean)/std
        
    # train_a = np.array([utils.pdSeries_to_npArray_singleMTS(train.iloc[i,:]) for i in range(train.shape[0])])
    # test_a  = np.array([utils.pdSeries_to_npArray_singleMTS(test.iloc[i,:]) for i in range(test.shape[0])])
    # for d in range(D):
    #     train_red_a = train_a[:,:,d]
    #     test_red_a  = test_a[:,:,d] 
    #     mean, std = np.nanmean(train_red_a), np.nanstd(train_red_a)
    #     print(mean, std)
    #     # replace
    #     train.iloc[:,d] = (train.iloc[:,d] - mean)/std
    #     test.iloc[:,d] =  (test.iloc[:,d] - mean)/std
    
    print('\nNormalized ; ', end=' ')
    return train, test

####### Interpolation ###########################################################################
def fill_interpolate(_train, _test, N_points):
    train = _train.copy()
    test  = _test.copy()
    interpolator = TSInterpolator(N_points)
    train = interpolator.fit_transform(train)
    test  = interpolator.transform(test)
    D = train.shape[1]
    for j in range(D):
        for i in range(train.shape[0]):     
            train.iloc[i,j] = pd.Series(train.iloc[i,j])
        for i in range(test.shape[0]):   
            test.iloc[i,j] = pd.Series(test.iloc[i,j])
    print('Interpolated to T = ', N_points)
    return train, test        

####### Clean ###########################################################################
def clean(_train, _test, labels_train, labels_test):
    train = _train.copy()
    test  = _test.copy()
    N_train, N_test = train.shape[0], test.shape[0]
    D = train.shape[1]
    i_dropped_train, i_dropped_test = [], []
    for i in range(N_train):
        Ts = []
        for d in range(D):
            Ts.append(len(train.iloc[i,d].index))
        if not  all(elem == Ts[0] for elem in Ts): 
            i_dropped_train.append(i)
    for i in range(N_test):
        Ts = []
        for d in range(D):
            Ts.append(len(test.iloc[i,d].index))
        if not  all(elem == Ts[0] for elem in Ts): 
            i_dropped_test.append(i)
    train = train.drop(i_dropped_train)
    test = test.drop(i_dropped_test)
    labels_train = np.delete(np.array(labels_train), i_dropped_train)
    labels_test = np.delete(np.array(labels_test), i_dropped_test)
    print('\nMTS with series with unequal length within different dimensions are removed')
    return train, test, labels_train, labels_test
    
####### Zero padding ########################################################################### 
def zero_padding(_train, _test, T_max):
    train = _train.copy()
    test  = _test.copy()
    print('Zero padding to T = ', T_max)
    N_train, N_test = train.shape[0], test.shape[0]
    D = train.shape[1]
    for i in range(N_train):
        T_diff = T_max - len(train.iloc[i,0])
        if T_diff>0:
            for d in range(D):
                series = pd.concat((train.iloc[i,d], pd.Series(np.zeros(T_diff))), ignore_index=True)
                train.iloc[i,d] = series
    for i in range(N_test):
        T_diff = T_max - len(test.iloc[i,0])
        if T_diff>0:
            for d in range(D):
                series = pd.concat((test.iloc[i,d], pd.Series(np.zeros(T_diff))), ignore_index=True)
                test.iloc[i,d] = series 
    return train, test    

def preprocessing(dataset_name, prepr_option, TRAIN_x_raw, TRAIN_y_raw, TEST_x_raw, TEST_y_raw, T_new=25, info=None):
    """############# Preprocessing #############################################################################"""
    # for each dataset, we subtract the mean and divide by the standard deviation (within the same dimensions)
    TRAIN_x, TEST_x = normalize_MTS(TRAIN_x_raw, TEST_x_raw) 
    
    # remove instances with dimensions of unequal lengths
    if dataset_name == 'CharacterTrajectories':
        TRAIN_x, TEST_x, TRAIN_y_raw, TEST_y_raw = clean(TRAIN_x, TEST_x, TRAIN_y_raw, TEST_y_raw)
    # Interpolate
    if prepr_option=='interpolate':
        # T_interp = 100
        # T_interp = math.ceil((T_max/math.ceil(T_max/25)))
        T_interp = T_new
        TRAIN_x, TEST_x = fill_interpolate(TRAIN_x, TEST_x, T_interp)    
    # Zero padding
    if prepr_option=='zero_padding':
        if not info[f'{dataset_name} train']['equalL'] or info[f'{dataset_name} test']['equalL']:
            TRAIN_x, TEST_x = zero_padding(TRAIN_x, TEST_x, T_new) 
    # Encoding for labels
    onehot_encoder = OneHotEncoder(sparse=False)
    TRAIN_y   = np.argmax(onehot_encoder.fit_transform(TRAIN_y_raw.reshape(-1,1)), axis=1)
    TEST_y    = np.argmax(onehot_encoder.transform(TEST_y_raw.reshape(-1,1)), axis=1)

    return TRAIN_x, TRAIN_y,  TEST_x, TEST_y