"""
BFSCPLATE2D - Implementation of the BFSC plate finite element in 2D

Author: Saullo G. P. Castro

"""
from .bfscplate2d import BFSCPlate2D, update_KC0, update_KCNL, update_KG, update_KG_cte_N, update_M, update_fint
from .bfscplate2d import INT, DOUBLE, KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE, KG_SPARSE_SIZE, M_SPARSE_SIZE
DOF = 10

