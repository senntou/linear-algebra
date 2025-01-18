from utils.cache import sqlite_cache
import numpy as np

from utils.input import get_input_images, get_input_images_except_all_zero


def get_covariance_matrix(data):
    return np.cov(data)


@sqlite_cache("eigen.db")
def get_eigen(cov):
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    # 固有値の大きい順に並び替え
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return eigvals, eigvecs


@sqlite_cache("eigvecs_of_lines.db")
def get_eigvals_of_lines(N, M):
    _params, data = get_input_images_except_all_zero(N, M)
    cov = get_covariance_matrix(data)
    eigvals, eigvecs = get_eigen(cov)
    return eigvals, eigvecs
