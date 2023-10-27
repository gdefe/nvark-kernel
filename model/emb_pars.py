import numpy as np
import matplotlib.pyplot as plt

# to set parameters
import cyl1tf
from scipy.signal import find_peaks












# ℓ₁ (L1) trend filtering algorithm developed by Kim et al. (2009)
def filter_MTS(_dataset, filter_scale, plot=False):
    dataset=_dataset.copy()
    N = len(dataset)
    D = dataset[0].shape[1]
    filtered_dataset = [np.zeros(dataset[i].shape) for i in range(N)]
    flag=0
    for i in range(N):
        for d in range(D):
            ts = dataset[i][:,d]
            fit = cyl1tf.calc_fit(ts, rel_scale=filter_scale)    
            filtered_dataset[i][:,d] = fit 
            if flag==0 and plot:
                plt.plot(ts)
                plt.plot(fit)
                plt.show()
                flag+=1
    return filtered_dataset

def time_between_peaks(dataset, T_min):
    mean_final = []
    shortest_final = []
    for i in range(len(dataset)):
        for d in range(dataset[i].shape[1]):
            ts = dataset[i][:,d]                
            # FIND PEAKS AND HOLLOWS
            peaks,   _ = find_peaks(ts)
            hollows, _ = find_peaks(-ts)
            crit_points = np.sort(np.concatenate((peaks,hollows)))
            
            # take the differences between peaks or hollow
            if len(crit_points)>1:
                dbp = crit_points[1:] - crit_points[:-1] 
                mean = int(np.nanmean(dbp))
                shortest = np.min(dbp)
                
            # else take just the first peak
            elif len(crit_points)==1:   
                mean     = min(T_min-2, crit_points[0])  
                shortest = min(T_min-2, crit_points[0]) 
                
            # else take just the length of the ts
            elif len(crit_points)==0:
                mean     = T_min-2
                shortest = T_min-2
                
            mean_final.append(mean)
            shortest_final.append(shortest)
        
        # # mean over dimensions
        # mean_final.append(np.nanmean(mean_D))
        # # min over dimensions
        # shortest_final.append(np.nanmin(shortest_D))  
    
    # mean over instances
    mean_over_i = np.nanmean(np.array(mean_final))
    # min over dimensions
    shortest_final = np.nanmin(shortest_final)
    return int(mean_over_i), int(shortest_final)
        
    # # find the time between peaks
    # t_w = utils.time_between_peaks(TRAIN_x_l, 0.001)
    # par_list = []
    # print('time_window = ', t_w)
    # for tbp in np.arange(t_w-3, t_w+3):
    #     # all pairs with given product
    #     num_arr = list(range(1, tbp+1))
    #     value_set = set(num_arr)
    #     par_list += [[n,int(tbp/n)] for n in num_arr if tbp/n in value_set]    
    
    
def SSR_parameters(TRAIN_x_l_, T_min,
                   filter_data = True, filter_scale = 0.1, 
                   plot=False):
    TRAIN_x_l = TRAIN_x_l_.copy()
    if filter_data:
        filtered_train = filter_MTS(TRAIN_x_l, 
                                          filter_scale=filter_scale, 
                                          plot=plot)
    else:
        filtered_train = TRAIN_x_l
    mean, shortest = time_between_peaks(filtered_train, T_min)
    print(f'mean = {mean} /// shortest = {shortest}')  
    s_short = shortest
    # s = int(round(np.sqrt(mean),0))
    s_sqrt = int(round(np.sqrt(mean),0))
    
    k_mean = int(np.round(mean,0))
    k_short = mean//shortest 
    # k = s
    k_sqrt = int(round(np.sqrt(mean),0))
    # print(f'k = {k_mean} /// s = {s_short}')  
    return k_sqrt, s_sqrt, k_mean, k_short, s_short