import numpy as np
import torch

def maximum_path(neg_cent, mask):
    """
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    mask = mask.data.cpu().numpy()
    
    path = np.zeros(neg_cent.shape, dtype=np.int32)
    
    t_t_max = mask.sum(1)[:, 0].astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].astype(np.int32)
    
    for b in range(neg_cent.shape[0]):
        path[b, :t_t_max[b], :t_s_max[b]] = maximum_path_each(neg_cent[b, :t_t_max[b], :t_s_max[b]])
        
    return torch.from_numpy(path).to(device=device, dtype=dtype)

def maximum_path_each(neg_cent):
    t_t, t_s = neg_cent.shape
    path = np.zeros((t_t, t_s), dtype=np.int32)
    v = np.full((t_t, t_s), -np.inf, dtype=np.float32)
    
    v[0, 0] = neg_cent[0, 0]
    for i in range(1, t_s):
        v[0, i] = v[0, i-1] + neg_cent[0, i]
        
    for i in range(1, t_t):
        for j in range(i, t_s):
            v[i, j] = max(v[i-1, j-1], v[i, j-1]) + neg_cent[i, j]
            
    curr_j = t_s - 1
    for i in range(t_t-1, -1, -1):
        path[i, curr_j] = 1
        if i > 0 and (curr_j == i or v[i-1, curr_j-1] >= v[i, curr_j-1]):
            curr_j -= 1
            
    return path

def mask_from_lens(lens, max_len=None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask
