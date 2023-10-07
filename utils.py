import numpy as np

def pdSeries_to_npArray_singleMTS(_MTS):
    # MTS is a row of a pdSeries DataFrame
    # obtained from MTS_datasets_pdS["key"].iloc[i,:]
    # has shape (D,)
    MTS = _MTS.copy()
    MTS_numpy = np.array([MTS[d] for d in range(len(MTS))]).T
    return MTS_numpy

def pdSeriesDataFrame_to_listOfnpArray(dataset):
    list_of_numpy = [pdSeries_to_npArray_singleMTS(dataset.iloc[i,:]) for i in range(len(dataset.index))]
    return list_of_numpy






def print_shape(nested_array):
    try: 
        nested_array.shape 
        if len(nested_array.shape)>1:
            return f'{nested_array.shape} -- 3D array'
    except: 
        N = len(nested_array)
        lengths = [nested_array[i].shape[0] for i in range(N)]
        if all(elem == lengths[0] for elem in lengths): T = f'{nested_array[0].shape[0]}'
        else: T = f'{max(lengths)}*'
        dims = [nested_array[i].shape[1] for i in range(N)]
        if all(elem == dims[0] for elem in dims): D = f'{nested_array[0].shape[1]}'
        else: D = f'{max(dims)}*'
        return f'([{N}], {T}, {D}) -- list 2D array' 