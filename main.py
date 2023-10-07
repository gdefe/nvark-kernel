import numpy as np
import time

from load_dataset import load_dataset, print_info, preprocessing

# NVARk
from model.NVARk import NVARk

# different tasks
from sklearn.svm import SVC

#internal imports
import utils
import tasks

datasets_list =  [  ##### ---univ---    
                    'SwedishLeaf',  
                    # 'CinCECGTorso',          
                   
                    # ##### ---multiv---
                    # 'JapaneseVowels',  
                    # 'UWaveGestureLibrary'
                    ]

general_setting = {'SwedishLeaf'          : {'k':4, 's':4},
                   'CinCECGTorso'         : {'k':9, 's':9},
                   'JapaneseVowels'       : {'k':2, 's':2},
                   'UWaveGestureLibrary'  : {'k':7, 's':7}}

"""global variables"""
# set to 'zero_padding' for matching the longest series in the dataset
# set to 'interpolate' 
prepr_option = 'zero_padding'     # 'none' / 'zero_padding' / 'interpolate'


def main():

      """################# Data Loading ##########################################"""
      for dataset_name in datasets_list:
            
            TRAIN_x_raw, TRAIN_y_raw, TEST_x_raw, TEST_y_raw = load_dataset(dataset_name)
            
            info = print_info(dataset_name, TRAIN_x_raw, TEST_x_raw, y=TRAIN_y_raw)
            if   prepr_option=='zero_padding': T_max = max(info[dataset_name+' train']['T_max'], info[dataset_name+' test']['T_max'])
            elif prepr_option=='interpolate' : T_max = 25
            
            """################# Preprocessing #################################"""
            TRAIN_x, TRAIN_y, TEST_x, TEST_y = preprocessing(dataset_name, prepr_option, 
                                                            TRAIN_x_raw, TRAIN_y_raw, TEST_x_raw, TEST_y_raw, 
                                                            T_new=T_max, info=info)   
            info = print_info(dataset_name, TRAIN_x, TEST_x, y=TRAIN_y)    
            print('\n')
            
            # convert datasets of panda series to a list of 2D numpy arrays (shape = [[N], T, D])
            TRAIN_x_l = utils.pdSeriesDataFrame_to_listOfnpArray(TRAIN_x)
            TEST_x_l  = utils.pdSeriesDataFrame_to_listOfnpArray(TEST_x)
                        
            
            """################# NVAR model ##########################################"""  
            params = {'k':general_setting[dataset_name]['k'], 
                        'n':2, 
                        's':general_setting[dataset_name]['s'], 
                        'n_dim':75, 
                        'lamb':None, 
                        'gamma_mult':1}   
            
            
            # uncomment one of the following
            
            
            """ individual steps to output K tr-tr """
            # model        = NVARk(**params, repr_mode='ridge', random_state=1, verbose_lvl=2)
            # _            = model.sample_indices(TRAIN_x_l)
            # R_nvar       = model.compute_embedding(TRAIN_x_l)
            # theta_repr   = model.linear_readout(R_nvar)
            # K            = model.rbf_function(theta_repr)
            
            
            
            """ RUNNING TIME: output matrices in one call and compute running time"""
            # st_time = time.perf_counter()
            # model      = NVARk(**params, repr_mode='ridge', random_state=1, verbose_lvl=2)
            # K_trtr     = model.compute_Ktrtr(TRAIN_x_l)
            # K_tetr     = model.compute_Ktetr(TRAIN_x_l, TEST_x_l)
            # end_time = time.perf_counter()
            # print('time = ', round(end_time - st_time,3), 's' )
            

            
            """ NVARk GENERAL SETTING """
            """ fix embedding parameters and fit SVM"""
            svm_C_list = np.logspace(-3, 3, 7)
            # mean over more iters
            random_iterations = 10
            accuracy=[]
            for i in range(1,random_iterations+1):
                  print(f'iteration {i}')
                  if i==1: verbose_lvl = 2
                  else:    verbose_lvl = 0
                  model = NVARk(**params, repr_mode='ridge', random_state=i, verbose_lvl=verbose_lvl, readout_type='SVM')
                  K_trtr     = model.compute_Ktrtr(TRAIN_x_l)
                  K_tetr     = model.compute_Ktetr(TRAIN_x_l, TEST_x_l)        
                  acc_test, acc_train, best_C = tasks.my_SVMopt_classifier(K_trtr, TRAIN_y, 
                                                                              K_tetr, TEST_y, 
                                                                              svm_C_list, i, n_folds=10, val_size=0.33, 
                                                                              verbose=False)
                  accuracy.append(acc_test) 
            print('accuracy = ', round(np.mean(accuracy),3) , ' +- ', round(np.std(accuracy),3))  



            """ NVARk* OPTIMIZED SETTING """
            """ optimize params via CV """
            # k_list = [1,2,3,4,5]
            # s_list = [1,2,3,4,5]
            # n_dim_list = [75]
            # svm_C_list = np.logspace(0,2,3)    # or np.logspace(-3,3,7)
            
            # model = NVARk(n=2, repr_mode='ridge', random_state=1, verbose_lvl=1, readout_type='SVM')
            
            # # optimize
            # st_time = time.perf_counter()
            # model.optimize_params(TRAIN_x_l, TRAIN_y, 
            #                       k_list, s_list, n_dim_list, svm_C_list, 
            #                       n_folds=10, val_size=0.33, n_jobs=-1)
            
            # # evaluate
            # # mean over more iters
            # random_iterations = 10
            # accuracy=[]
            # for i in range(1,random_iterations+1):
            #     print(f'iteration {i}')
            #     model.random_state = i
            #     if i==1:model.verbose_lvl = 2
            #     else:   model.verbose_lvl = 0
            #     model.fit(TRAIN_x_l, TRAIN_y)
            #     accuracy.append(model.score(TEST_x_l, TEST_y))
            # end_time = time.perf_counter()
            # print('time = ', round(end_time - st_time,3), 's' )
            # print('accuracy = ', round(np.mean(accuracy),3) , ' +- ', round(np.std(accuracy),3))







if __name__ == "__main__":
    main()