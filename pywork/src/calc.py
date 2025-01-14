from cache import sqlite_cache
import numpy as np

from input import get_input_images


def get_covariance_matrix(data):
    return np.cov(data)


@sqlite_cache("eigen.db")
def get_eigen(cov):
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    return eigvals, eigvecs


@sqlite_cache("eigvecs_of_lines.db")
def get_eigvals_of_lines(N, M):
    data = get_input_images(N, M)
    cov = get_covariance_matrix(data)
    eigvals, eigvecs = get_eigen(cov)
    return eigvals, eigvecs
