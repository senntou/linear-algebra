from cache import sqlite_cache
import numpy as np


def get_covariance_matrix(data):
    return np.cov(data)


@sqlite_cache("eigen.db")
def get_eigen(cov):
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    return eigvals, eigvecs
