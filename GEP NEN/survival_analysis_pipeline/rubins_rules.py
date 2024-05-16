import numpy as np
import warnings

def pool_single_mean(parameter):
    """Compute the pooled mean of a parameter.
    
    Parameters
    ----------
    parameter : array-like of shape (M,)
        The parameter values from M imputed datasets.
    
    Returns
    -------
    float : pooled mean"""

    # check if parameter is numpy. If not, convert it to numpy
    if not isinstance(parameter, (np.ndarray, list)):
        warnings.warn("Expected list or array-like datatype.", UserWarning)
        return 
    
    parameter = np.array(parameter)

    # check if parameter is empty
    if parameter.size == 0:
        warnings.warn("The parameter array is empty.", RuntimeWarning)
        return 
    
    return parameter.mean(axis=0)

def pool_multiple_means(parameters):
    """Compute the pooled means of multiple parameters.
    
    Parameters
    ----------
    parameter : array-like of shape (M,d)
        The d parameter values from M imputed datasets.
    
    Returns
    -------
    array : array-like of shape (d,)
        The pooled mean of each parameter
    """

    # check if parameter is numpy. If not, convert it to numpy
    if not isinstance(parameters, (np.matrix, list)):
        warnings.warn("Expected matrix or nested list.", UserWarning)
        return
    
    parameters = np.matrix(parameters)

    # check if parameters is empty
    if parameters.size == 0:
        warnings.warn("The parameter matrix is empty.", RuntimeWarning)
        return
    
    return np.array(parameters.mean(axis=0)).flatten()

def pool_single_var(parameter, parameter_var):
    """Compute the pooled variance of a parameter.
    
    Parameters
    ----------
    parameter : array-like of shape (M,)
        The parameter values from M imputed datasets.

    parameter_var : array-like of shape (M,)
        The parameter variance in each of the M imputed datasets.

    Returns
    -------
    float : pooled variance"""

    if not (isinstance(parameter, (np.ndarray, list)) and isinstance(parameter_var, (np.ndarray, list))):
        warnings.warn("Expected list or array-like datatype.", UserWarning)
        return 
    
    parameter = np.array(parameter)
    parameter_var = np.array(parameter_var)

    if (parameter.size == 0) and (parameter_var.size == 0):
        warnings.warn("At least one of the parameter arrays are empty.", RuntimeWarning)
        return None
    
    m = len(parameter)
    # Within-imputation variance
    W =  parameter_var.mean(axis=0)

    # Between-imputation variance
    mean = pool_single_mean(parameter)
    diff = (parameter - mean)**2
    B = diff.var(axis=0, ddof=1)

    # Combine
    var = W + (1+1/m)*B
    return var

def pool_multiple_vars(parameters, parameter_vars):
    """Compute the pooled variances of multiple parameters.
    
    Parameters
    ----------
    parameters : array-like of shape (M,d)
        The parameter values from M imputed datasets.

    parameter_vars : array-like of shape (M,d)
        The parameter variance in each of the M imputed datasets.

    Returns
    -------
    array : array-like of shape (d,)
        Pooled variance"""

    if not (isinstance(parameters, (np.matrix, list)) and isinstance(parameter_vars, (np.matrix, list))):
        warnings.warn("Expected matrix or nested list.", UserWarning)
        return 
    
    parameters = np.matrix(parameters)
    parameter_vars = np.matrix(parameter_vars)
    
    if (parameters.size == 0) or (parameter_vars.size == 0):
        warnings.warn("At least one of the parameter arrays are empty.", RuntimeWarning)
        return None
    
    m = parameters.shape[0]
    # Within-imputation variance
    W =  parameter_vars.mean(axis=0)

    # Between-imputation variance
    means = pool_multiple_means(parameters)
    diff = np.square(parameters-means)
    B = diff.var(axis=0, ddof=1)

    # Combine
    var = np.array(W + (1+1/m)*B).flatten()
    return var

if __name__ == "__main__":
    # Example usage
    parameter = [1, 2, 2, 2, 1]
    parameter_var = [0.2, 0.1, 0.2, 0.1, 0.3]

    print("Single pooling")
    print(pool_single_mean(parameter))
    print(pool_single_var(parameter, parameter_var))

    parameters = [[1,2,3],
                 [4,2,2]]
    parameter_vars = [[0.2, 0.3, 0.3],
                     [1, 0.2, 0.1]]
    
    print("Multiple pooling")
    print(pool_multiple_means(parameters))
    print(pool_multiple_vars(parameters, parameter_vars))

    # Raises error because incorrect input type.
    #not_list = {"test": [1,2,3]}
    #print(pool_single_mean(not_list))
    #print(pool_multiple_means(not_list))
    #print(pool_single_var(parameters, not_list))
    #print(pool_multiple_vars(not_list, parameter_vars))
    