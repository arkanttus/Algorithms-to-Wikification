from numpy import ravel_multi_index

def sub2ind(arr_shape, rows, cols):
    return rows + (cols-1) * arr_shape[0]
    #return ravel_multi_index( ind, arr_shape, order='F' )