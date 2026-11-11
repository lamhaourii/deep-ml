import numpy as np

def add_positional_embeddings(token_embeddings: np.ndarray, pos_embedding_matrix: np.ndarray) -> np.ndarray:
    """Add absolute positional embeddings to token embeddings."""
    b, seq_len, e_dim = token_embeddings.shape

    return token_embeddings + pos_embedding_matrix[:seq_len, :]

