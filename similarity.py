# similarity.py

from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(query_feats, gallery_feats):
    """
    Computes cosine similarity between a query vector (or batch)
    and all gallery vectors.
    
    Parameters:
        query_feats: numpy array of shape (n_query, feature_dim)
        gallery_feats: numpy array of shape (n_gallery, feature_dim)
        
    Returns:
        similarity_matrix: numpy array of shape (n_query, n_gallery)
    """
    return cosine_similarity(query_feats, gallery_feats)
