import numpy as np
from numba import jit

@jit(nopython=True)
def maximum_path_c(path, neg_cent, t_t_max, t_s_max):
  for b in range(path.shape[0]):
    t_t = t_t_max[b]
    t_s = t_s_max[b]
    
    v = np.empty((t_t, t_s), dtype=np.float32)
    v[0, 0] = neg_cent[b, 0, 0]
    for i in range(1, t_s):
      v[0, i] = v[0, i-1] + neg_cent[b, 0, i]
      
    for i in range(1, t_t):
      for j in range(i, t_s):
        v_prev_diag = v[i-1, j-1]
        v_prev_row = v[i, j-1]
        if v_prev_diag > v_prev_row:
          v[i, j] = v_prev_diag + neg_cent[b, i, j]
        else:
          v[i, j] = v_prev_row + neg_cent[b, i, j]
          
    curr_j = t_s - 1
    for i in range(t_t-1, -1, -1):
      path[b, i, curr_j] = 1
      if i > 0:
        if curr_j == i or v[i-1, curr_j-1] >= v[i, curr_j-1]:
          curr_j -= 1
