import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    return X@W_q, X@W_k, X@W_v

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k=Q.shape[-1]
    attn = Q@K.swapaxes(-1, -2)/np.sqrt(d_k)
    max_attn=np.max(attn, axis=-1, keepdims=True)

    score = np.exp(attn-max_attn)/np.sum(np.exp(attn-max_attn),axis=-1, keepdims=True)
    return score @ V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    T, d_model=Q.shape
    d_k=d_model//n_heads
    QQ = Q.reshape(T,n_heads,d_k).transpose(1,0,2)
    KK = K.reshape(T,n_heads,d_k).transpose(1,0,2)
    VV = V.reshape(T,n_heads,d_k).transpose(1,0,2)
    A=self_attention(QQ, KK, VV)
    return A.transpose(1,0,2).reshape(T,-1)