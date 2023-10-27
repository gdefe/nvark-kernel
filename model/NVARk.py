# general imports
import numpy as np
# import pandas as pd
import sklearn as sk
from sklearn.base import BaseEstimator
import itertools 
import random
import warnings

# linear readout
import scipy
from sklearn.linear_model import Ridge

# rbf function
from scipy.spatial.distance import pdist, cdist, squareform

# different tasks
from sklearn.svm import SVC

# CV
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import ParameterGrid

# internal imports
import utils



def apply_lag(series, n):
    """ apply a lag to an univariate of multivariate series 
    
    Parameters:
        series    (T x D) array:  input series
        n         (int):   lag size
                
    Returns:
        e       (T x D) array:  lagged series
    """  
    e = np.empty_like(series)
    if n > 0:
        e[:n] = np.nan
        e[n:] = series[:-n]
    elif n==0: 
        e = series
    else:
        e[n:] = np.nan
        e[:n] = series[-n:]
    return e  




class NVARk(BaseEstimator):

    def __init__(self, k=1, n=2, s=1, n_dim=75, lamb=None, gamma_mult=1,
                 repr_mode='ridge', readout_type=None, svm_C=1,
                 random_state=1, verbose_lvl=0,
                 ):
        """ Initialize the NVAR model, storing parameters and configurations
        
        Parameters:
            k            (int):    number of lags
            n            (int):    polynomial order for nonlinear functionals
            s            (int):    lag size
            n_dim        (int):    maximum dimensionality of the NVAR embedding
            lamb         (float):  regularization of the readout ridge regression fit; if None (default), the OCReP algorithm is used 
            gamma_mult   (float):  scaling for the lengthscale of the rbf function
            repr_mode    (string): only implemented as 'ridge', one can define different readout modules
            readout_type (string): '1NN'/'SVM' or None; define the end task
            svm_C        (float):  parameter of the SVM
            random_state
            verbose_lvl
        """  
        # embedding parameters
        self.k = k
        self.n = n
        self.s = s
        self.n_dim = n_dim
        
        # representation parameters
        self.repr_mode = repr_mode
        self.lamb = lamb
        self.gamma_mult = gamma_mult
        
        # random seed
        self.random_state = random_state
        # verbose level
        self.verbose_lvl = verbose_lvl
        if self.verbose_lvl==2: 
            print('--NVARk model--')
            print(f'\tinit params = [k:{self.k}, n:{self.n}, s:{self.s}, d_red:{self.n_dim}, lamb:{self.lamb}, gamma_mult:{self.gamma_mult}]')
        
        # readout
        self.readout_type = readout_type
        self.svm_C = svm_C
        
        self.theta_repr_tr = None
        return 
    
    
    def sample_indices(self, DATA_x_l):
        """ Sample the selected dimensions in the NVAR embedding
        
        Parameters:
            DATA_x_l    (list of 2D arrays):  input data
        
        Returns:
            sampled_indices (list of indices)  
        """  
        # input data, shape [[N], T, D] or pd.DataFrame
        if type(DATA_x_l)==list: self.D = DATA_x_l[0].shape[1]  
        # elif type(DATA_x_l)==pd.DataFrame: self.D = len(self.X.columns)
        else: raise RuntimeError('Invalid input type. Must be list of 2D arrays [[N], T, D]')
        
        # sample indices
        n_dim_add = self.n_dim - self.D          # calculate the number of dimensions to add to reach n_dim
        if n_dim_add <= 0: 
            warnings.warn('The dimensionality of the input is larger than the maximum allowed dimension for the embedding. \
                           the execution will continue simply copying the input into the embedding. \
                           Please consider increasing the "n_dim" parameter in the constructor')
            self.sampled_indices = []
        else:    
            # create the list with all possible dimensions possibilities
            i_lagged    = list(range(self.D*self.k))          # indices for lagged dimensions
            i_linear    = list(range(self.D*(self.k+1)))      # indices for lagged dimensions + input series
            if self.n > 1: i_nonlinear = list(itertools.combinations_with_replacement(range(len(i_linear)), self.n))      # indices for polynomial combinations of all linear terms
            else: i_nonlinear = []
            # possible pool of combinations to add
            # lagged variables are indicated by a single index, polynomial combinations are indicatedby a list of factors
            i_total = i_lagged + i_nonlinear
            
            self.sampled_indices = self.maybe_sample_index(n_dim_add, i_total)
            self.sampled_lag_indices = self.maybe_sample_index(n_dim_add, i_lagged)
            self.sampled_nonlin_indices = self.maybe_sample_index(n_dim_add, i_nonlinear)        
        return self.sampled_indices


    def maybe_sample_index(self, n_dim_add, indices):
        if len(indices)<n_dim_add: 
            # the possible concatenations are less than n_dim
            sampled_indices = indices 
        else:
            # sample from total
            random.seed(self.random_state)
            sampled_indices = random.sample(indices, n_dim_add)
        return sampled_indices
        
    def compute_embedding(self, DATA_x_l, indices=None):
        """ compute the NVAR embedding, given a list of time series and indices to 
        asses which dimensions and combinations should be considered
        
        Parameters:
            DATA_x_l    (list of 2D arrays):    input data
            indices     (list of indices) : a list can be provided if not already computed
        Returns:
            R_nvar     (list of 2D arrays ): NVAR delay-embedding  
        """ 
        if self.verbose_lvl==2: print('\tNVAR embeddings computation...')
        # to store the embedding states 
        if self.verbose_lvl==2: print("\t\tinput shape: \t", utils.print_shape(DATA_x_l))
        N = len(DATA_x_l)
        R_nvar = DATA_x_l.copy()
        lagged_terms = []
        linear_terms = DATA_x_l.copy()
        
        self.n_drop = (self.k)*self.s 
        
        ########## compute the nvar states ##########
        # concatenate additional dimensions to the input MTS   
        for i in range(N):
            # create all linear D(k+1) terms with lags
            for k in range(1, self.k+1):
                lag = k*self.s
                # shift and concatenate, create nans on top of the shifted values
                linear_terms[i] = np.concatenate((linear_terms[i], apply_lag(DATA_x_l[i],lag)), axis=1)
            # the lagged terms are the linear ones without the input series
            lagged_terms.append( linear_terms[i][:,self.D:] )
            
            # concatenate 
            for index in self.sampled_indices:
                R_nvar[i] = self.cat_dimension(index, R_nvar[i], lagged_terms[i], linear_terms[i])
            # for index in self.sampled_lag_indices:
            #     R_nvar[i] = self.cat_dimension(index, R_nvar[i], lagged_terms[i], linear_terms[i])
            # for index in self.sampled_nonlin_indices:
            #     if R_nvar[i].shape[-1] >= self.n_dim: break
            #     R_nvar[i] = self.cat_dimension(index, R_nvar[i], lagged_terms[i], linear_terms[i])
                    
            # drop first states
            # resultant series will be empty if n_drop > length of the series
            R_nvar[i] = R_nvar[i][self.n_drop:,:]   
            
        if self.verbose_lvl==2: print("\t\tR_nvar shape: \t", utils.print_shape(R_nvar)) 
        return R_nvar
            
        

    def cat_dimension(self, index, block, lagged_term, linear_term):
        # linear terms
        if type(index)==int: 
            block = np.concatenate((block, 
                                    lagged_term[:,index].reshape(-1,1)), 
                                    axis=1)
        # quadratic terms
        if type(index)==tuple and self.n==2:
            block = np.concatenate((block, 
                                    (linear_term[:,index[0]]*linear_term[:,index[1]]).reshape(-1,1)),
                                    axis=1)
        return block      
    
    
    
    
    def linear_readout(self, R_nvar, thr=10**-20):
        """ apply the linear readout to given embeddings
        
        Parameters:
            R_nvar     (list of 2D arrays ): NVAR delay-embedding  
        Returns:
            input_repr  (2D array):  all representation vectors
        """ 
        N = len(R_nvar)
        D = R_nvar[0].shape[1]
        if self.repr_mode=='ridge':
            coeff_tr  = []
            biases_tr = []
            # fit ridge regression           
            for i in range(N):
                # check if series is empy after the dropout
                if R_nvar[i].shape[0] > 1:
                    if self.lamb is None: 
                        # OCRep regularization optimization
                        s_vals = scipy.linalg.svdvals(R_nvar[i])
                        s_max = s_vals[0]
                        s_min = [val for val in s_vals if val>thr][-1]
                        OCReP_reg = s_max*s_min
                        # print(i, ' ' ,OCReP_reg)
                        if OCReP_reg < thr: 
                            print(i, ' ' , s_vals)
                            raise RuntimeError('singular matrix')
                            # lamb=1
                        lamb = OCReP_reg
                    else: 
                        lamb = self.lamb 
                    self._ridge_embedding = Ridge(alpha=lamb, fit_intercept=True)
                    self._ridge_embedding.fit(R_nvar[i][0:-1, :], R_nvar[i][1:, :])     # fit to next embedding state
                    coeff_tr.append(self._ridge_embedding.coef_.ravel())
                    biases_tr.append(self._ridge_embedding.intercept_.ravel())
                else:
                    coeff_nans  = np.empty((D,D)).ravel(); coeff_nans[:] = np.nan
                    biases_nans = np.empty(D).ravel(); biases_nans[:] = np.nan   
                    coeff_tr.append(coeff_nans)
                    biases_tr.append(biases_nans)
            input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)     # concatenate
        else:
            raise RuntimeError('Invalid representation mode')
        if self.verbose_lvl==2:  print("\t\trep shape: \t\t", input_repr.shape, ' -- 2D array')
        return input_repr
    
    
    
    def rbf_function(self, tr_rep, te_rep=None, mode='tr-tr'):
        """ apply the RBF function to computed representations
        
        Parameters:
            tr_rep     (2D array):  all training representation vectors
            te_rep     (2D array):  all test representation vectors
            mode       (string):   'tr-tr'/ 'te-tr'; select the desired kernel matrix to compute 
        Returns:
            Kernel_matrix  (2D array):  kernel matrix
        """ 
        if mode=='tr-tr':
            # pairwise distance
            pdistance = squareform(pdist(tr_rep, metric='euclidean')) 
            # RBF gamma median estimator
            self.RBF_gamma = self.gamma_mult * np.nanmedian(pdistance)
            # Kernel
            if self.RBF_gamma==0 : self.RBF_gamma=1 # fix division by zero
            Kernel_matrix = np.exp( -(pdistance)**2 / (2*self.RBF_gamma**2))
            
        elif mode=='te-tr':
            assert(te_rep is not None)
            # pairwise distance
            pdistance = cdist(te_rep, tr_rep, metric='euclidean')
            # Kernel
            Kernel_matrix = np.exp( -(pdistance)**2 / (2*self.RBF_gamma**2))
            
        elif mode=='te-te':
             assert(te_rep is not None)
             # pairwise distance
             pdistance = cdist(te_rep, te_rep, metric='euclidean')
             # Kernel
             Kernel_matrix = np.exp( -(pdistance)**2 / (2*self.RBF_gamma**2))
        return Kernel_matrix
    
    
    
    def compute_Ktrtr(self, train):
        """ Given input data, compute the train-train kernel matrix
        
        Parameters:
            train     (list of 2D arrays):    input train data 
        Returns:
            Kernel_matrix  (2D array):  train-train kernel matrix
        """ 
        _                  = self.sample_indices(train)
        self.R_nvar_tr     = self.compute_embedding(train)
        self.theta_repr_tr = self.linear_readout(self.R_nvar_tr)
        self.K_trtr        = self.rbf_function(self.theta_repr_tr, None, 'tr-tr')
        # if np.isnan(self.K_trtr).any():
        #     warnings.warn('nan in the output matrix. Check the length of the time series after the dropout')
        return self.K_trtr
    
    
    
    def compute_Ktetr(self, test, train=None): 
        """ Given input data, compute the test-train kernel matrix
        
        Parameters:
            test      (list of 2D arrays):    input test data 
            train     (list of 2D arrays):    input train data 
        Returns:
            Kernel_matrix  (2D array):  test-train kernel matrix
        """ 
        if self.theta_repr_tr is None:
            if train is None:
                raise ValueError('train input must be given if "compute_Ktrtr" has not been called before')
            _                  = self.sample_indices(train)
            self.R_nvar_tr     = self.compute_embedding(train)
            self.theta_repr_tr = self.linear_readout(self.R_nvar_tr)
            self.RBF_gamma = self.gamma_mult * np.median(squareform(pdist(self.theta_repr_tr, metric='euclidean')) )
            
        self.R_nvar_te      = self.compute_embedding(test)
        self.theta_repr_te  = self.linear_readout(self.R_nvar_te)
        self.K_tetr         = self.rbf_function(self.theta_repr_tr, self.theta_repr_te, 'te-tr')  
        # if np.isnan(self.K_tetr).any():
        #     warnings.warn('nan in the output matrix. Check the length of the time series after the dropout')
        return self.K_tetr


    def compute_Ktete(self, test=None, train=None): 
        """ Given input data, compute the test-train kernel matrix
        
        Parameters:
            test      (list of 2D arrays):    input test data 
            train     (list of 2D arrays):    input train data 
        Returns:
            Kernel_matrix  (2D array):  test-train kernel matrix
        """ 
        if self.theta_repr_tr is None:
            if train is None:
                raise ValueError('train input must be given if "compute_Ktrtr" has not been called before')
            _                  = self.sample_indices(train)
            self.R_nvar_tr     = self.compute_embedding(train)
            self.theta_repr_tr = self.linear_readout(self.R_nvar_tr)
            self.RBF_gamma = self.gamma_mult * np.median(squareform(pdist(self.theta_repr_tr, metric='euclidean')) )
            
        if self.theta_repr_te is None:
            if test is None:
                raise ValueError('test input must be given if "compute_Ktetr" has not been called before')
            self.R_nvar_te      = self.compute_embedding(test)
            self.theta_repr_te  = self.linear_readout(self.R_nvar_te)
        self.K_tete         = self.rbf_function(self.theta_repr_te, self.theta_repr_te, 'te-te')  
        return self.K_tete




    def fit(self, TRAIN_x_l, TRAIN_y=None):
        """ Compute the train-train kernel matrix and fit the desired end task machine
        
        Parameters:
            TRAIN_x_l (list of 2D arrays):    input train data 
            TRAIN_y   (list):                 input train labels
        """ 
        self.TRAIN_x_l = TRAIN_x_l
        
        # compute Ktrtr
        if self.verbose_lvl==2: print('\tcomputing K tr-tr...')
        self.Ktrtr = self.compute_Ktrtr(TRAIN_x_l) 
        
        if not np.isnan(self.Ktrtr).any():
            if self.readout_type=='SVM':
                self.svm = SVC(C=self.svm_C, kernel='precomputed', class_weight=None, decision_function_shape='ovr', cache_size = 500) 
                self.svm.fit(self.Ktrtr, TRAIN_y)
                
            elif self.readout_type=='1NN':
                self.TRAIN_y = TRAIN_y
                
            elif self.readout_type==None:
                return self.Ktrtr
            else:
                raise RuntimeError('Invalid readout type')
                
        return self
            





    def predict(self, TEST_x_l):
        """ Compute the test-train kernel matrix and output labels for the test set
        
        Parameters:
            TEST_x_l (list of 2D arrays):     input test data 
        
        Returns:
            output   (list):                 predicted labels
        """ 
        self.TEST_x_l = TEST_x_l
        # computing K te-tr
        if self.verbose_lvl==2: print('\tcomputing K te-tr...')
        self.Ktetr = self.compute_Ktetr(self.TEST_x_l, self.TRAIN_x_l)         
        
        if not np.isnan(self.Ktetr).any():
            if self.readout_type=='SVM':
                self.output = self.svm.predict(self.Ktetr)
                
            elif self.readout_type=='1NN':
                self.output = []
                for i in range(self.Ktetr.shape[0]):
                    sim = self.Ktetr[i,:]
                    j = np.argmax(sim)
                    self.output.append(self.TRAIN_y[j])
            elif self.readout_type==None:
                return self.Ktrtr
            else:
                raise RuntimeError('Invalid readout type')
        else:
            self.output = None
            
        return self.output
        
    
    
    def score(self, TEST_x_l, TEST_y):
        """ Predict the test labels and compute the classification accuracy
        
        Parameters:
            TEST_x_l (list of 2D arrays):     input test data 
            TEST_y   (list):                  ground truth test labels
        Returns:
            accuracy (float):                 classification accuracy
        """ 
        if not np.isnan(self.Ktrtr).any():
            labels_pred = self.predict(TEST_x_l)
            if labels_pred is not None:
                accuracy = np.sum(labels_pred == TEST_y)/len(labels_pred)
            else: accuracy = 0.
        else: accuracy = 0.
        return accuracy


    def get_params(self, deep=True):
        return {"k":        self.k, 
                "n":        self.n,
                "s":        self.s,
                "n_dim":    self.n_dim,
                "lamb":     self.lamb,
                "gamma_mult": self.gamma_mult,
                
                "svm_C":           self.svm_C,
                "repr_mode":       self.repr_mode, 
                "readout_type":    self.readout_type,
                "random_state":    self.random_state,
                "verbose_lvl":     self.verbose_lvl}


    def optimize_params(self, TRAIN_x_l, TRAIN_y,
                        k_list=None, s_list=None, n_dim_list=None, svm_C_list=None,      
                        n_folds=10, val_size=0.33, n_jobs=1, random_state=1234,
                        split='stratified'):
        """ Optimize parameters of an initialized model
        
        Parameters:
            TRAIN_x_l (list of 2D arrays):     input test data 
            TRAIN_y   (list):                  ground truth train labels
            k_list    (list):                  CV grid for k
            s_list    (list):                  CV grid for s
            n_dim_list    (list):              CV grid for n_dim
            svm_C_list    (list):              CV grid for 
            n_folds       (int):               number of folds in the CV optimization
            val_size     (float):              percentage of the train test to be used as validation set
            n_jobs        :                    
        """ 
        params = {'k':     k_list if k_list is not None else [self.k],
                  's':     s_list if s_list is not None else [self.s],
                  'n_dim': n_dim_list if n_dim_list is not None else [self.n_dim], 
                  'svm_C': svm_C_list if svm_C_list is not None else [self.svm_C], 
                  }
        param_grid = list(ParameterGrid(params))
        if split=='stratified':
            sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=val_size, random_state=random_state)
        elif split=='random':
            sss = ShuffleSplit(n_splits=n_folds, test_size=val_size, random_state=random_state)
        else:
            raise RuntimeError('Invalid split type')
        if self.verbose_lvl==1:
            print(f'\tNVARk CV optimization with n_folds={n_folds} and val_size={val_size}')  
            print(f'\tgrid = {params}')
            print('\tlen of param_grid: ', len(param_grid))
        gs = sk.model_selection.GridSearchCV(self,                           
                                             cv=sss, 
                                             param_grid=params,
                                               # scoring='accuracy',
                                             scoring=None,
                                             n_jobs=n_jobs)
        gs.fit(TRAIN_x_l, TRAIN_y) 
        if self.verbose_lvl==1:
            print('\tbest parameters:', gs.best_params_)
            print('\tbest score on validation set:', gs.best_score_)
        best_pars = gs.best_params_
        self.set_params(**best_pars)
        
        
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self