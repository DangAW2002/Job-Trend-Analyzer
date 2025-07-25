"""
Clustering Module
Gom cụm các embedding để tìm ra các nhóm kỹ năng/xu hướng tương tự
"""

import logging
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClusterResult:
    """Data class for clustering results"""
    cluster_id: int
    items: List[str]
    scores: List[float]
    embeddings: List[List[float]]
    centroid: Optional[List[float]] = None
    size: int = 0
    avg_score: float = 0.0
    
    def __post_init__(self):
        self.size = len(self.items)
        self.avg_score = np.mean(self.scores) if self.scores else 0.0
        if self.embeddings and not self.centroid:
            self.centroid = np.mean(self.embeddings, axis=0).tolist()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'cluster_id': self.cluster_id,
            'items': self.items,
            'scores': self.scores,
            'size': self.size,
            'avg_score': self.avg_score,
            'centroid': self.centroid
        }

class EmbeddingClusterer:
    """Class để gom cụm các embedding"""
    
    def __init__(self,
                 algorithm: str = 'kmeans',
                 n_clusters: int = 10,
                 random_state: int = 42,
                 normalize_embeddings: bool = True,
                 **kwargs):
        """
        Initialize clustering algorithm
        
        Args:
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Number of clusters (for algorithms that need it)
            random_state: Random state for reproducibility
            normalize_embeddings: Whether to normalize embeddings before clustering
            **kwargs: Additional parameters for clustering algorithms
        """
        self.algorithm = algorithm.lower()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.normalize_embeddings = normalize_embeddings
        self.kwargs = kwargs
        
        # Initialize scaler
        self.scaler = StandardScaler() if normalize_embeddings else None
        
        # Initialize clustering model
        self.model = self._initialize_model()
        self.is_fitted = False
        
        logger.info(f"✅ Initialized {algorithm} clusterer with {n_clusters} clusters")
    
    def _initialize_model(self):
        """Initialize the clustering model based on algorithm"""
        if self.algorithm == 'kmeans':
            return KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                **self.kwargs
            )
        elif self.algorithm == 'dbscan':
            return DBSCAN(
                eps=self.kwargs.get('eps', 0.5),
                min_samples=self.kwargs.get('min_samples', 5),
                **{k: v for k, v in self.kwargs.items() if k not in ['eps', 'min_samples']}
            )
        elif self.algorithm == 'hierarchical':
            return AgglomerativeClustering(
                n_clusters=self.n_clusters,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def _prepare_embeddings(self, embeddings: List[List[float]]) -> np.ndarray:
        """Prepare embeddings for clustering"""
        X = np.array(embeddings)
        
        if self.normalize_embeddings and self.scaler:
            X = self.scaler.fit_transform(X)
        
        return X
    
    def fit_predict(self, embeddings: List[List[float]]) -> np.ndarray:
        """
        Fit the model and predict cluster labels
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Array of cluster labels
        """
        X = self._prepare_embeddings(embeddings)
        
        try:
            labels = self.model.fit_predict(X)
            self.is_fitted = True
            logger.info(f"✅ Clustering completed. Found {len(set(labels))} clusters")
            return labels
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise
    
    def evaluate_clustering(self, embeddings: List[List[float]], labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality
        
        Args:
            embeddings: List of embedding vectors
            labels: Cluster labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        X = self._prepare_embeddings(embeddings)
        
        metrics = {}
        
        try:
            # Silhouette score (higher is better, range: -1 to 1)
            if len(set(labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(X, labels)
            else:
                metrics['silhouette_score'] = 0.0
            
            # Calinski-Harabasz score (higher is better)
            if len(set(labels)) > 1:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            else:
                metrics['calinski_harabasz_score'] = 0.0
            
            # Inertia (for KMeans, lower is better)
            if hasattr(self.model, 'inertia_'):
                metrics['inertia'] = self.model.inertia_
            
            # Number of clusters
            metrics['n_clusters'] = len(set(labels))
            
            # Cluster sizes
            unique_labels, counts = np.unique(labels, return_counts=True)
            metrics['cluster_sizes'] = dict(zip(unique_labels.tolist(), counts.tolist()))
            
        except Exception as e:
            logger.warning(f"Could not calculate all metrics: {e}")
        
        return metrics
    
    def cluster_embeddings(self, 
                          items: List[str],
                          scores: List[float], 
                          embeddings: List[List[float]]) -> List[ClusterResult]:
        """
        Cluster embeddings and return organized results
        
        Args:
            items: List of item names (e.g., n-grams)
            scores: List of scores for each item
            embeddings: List of embedding vectors
            
        Returns:
            List of ClusterResult objects
        """
        if len(items) != len(scores) or len(items) != len(embeddings):
            raise ValueError("Items, scores, and embeddings must have the same length")
        
        if not embeddings:
            return []
        
        # Perform clustering
        labels = self.fit_predict(embeddings)
        
        # Evaluate clustering
        metrics = self.evaluate_clustering(embeddings, labels)
        logger.info(f"📊 Clustering metrics: {metrics}")
        
        # Organize results by cluster
        cluster_dict = {}
        for i, label in enumerate(labels):
            if label not in cluster_dict:
                cluster_dict[label] = {
                    'items': [],
                    'scores': [],
                    'embeddings': []
                }
            
            cluster_dict[label]['items'].append(items[i])
            cluster_dict[label]['scores'].append(scores[i])
            cluster_dict[label]['embeddings'].append(embeddings[i])
        
        # Create ClusterResult objects
        results = []
        for cluster_id, data in cluster_dict.items():
            result = ClusterResult(
                cluster_id=int(cluster_id),
                items=data['items'],
                scores=data['scores'],
                embeddings=data['embeddings']
            )
            results.append(result)
        
        # Sort by average score (descending)
        results.sort(key=lambda x: x.avg_score, reverse=True)
        
        return results
    
    def find_optimal_clusters(self, 
                            embeddings: List[List[float]], 
                            min_clusters: int = 2,
                            max_clusters: int = 20) -> Dict[int, Dict[str, float]]:
        """
        Find optimal number of clusters using various metrics
        
        Args:
            embeddings: List of embedding vectors
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Dictionary mapping n_clusters to metrics
        """
        if self.algorithm != 'kmeans':
            logger.warning("Optimal cluster finding is only implemented for KMeans")
            return {}
        
        X = self._prepare_embeddings(embeddings)
        results = {}
        
        logger.info(f"🔍 Finding optimal clusters in range {min_clusters}-{max_clusters}")
        
        for n in range(min_clusters, max_clusters + 1):
            try:
                # Create temporary KMeans model
                temp_model = KMeans(n_clusters=n, random_state=self.random_state, n_init=10)
                labels = temp_model.fit_predict(X)
                
                metrics = {
                    'silhouette_score': silhouette_score(X, labels),
                    'inertia': temp_model.inertia_,
                    'calinski_harabasz_score': calinski_harabasz_score(X, labels)
                }
                
                results[n] = metrics
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {n} clusters: {e}")
        
        return results

# Convenience functions
def cluster_ngrams(ngrams_with_embeddings: List[Tuple[str, float, List[float]]], 
                  n_clusters: int = 10,
                  algorithm: str = 'kmeans') -> List[ClusterResult]:
    """
    Quick function to cluster n-grams with embeddings
    
    Args:
        ngrams_with_embeddings: List of (ngram, score, embedding) tuples
        n_clusters: Number of clusters
        algorithm: Clustering algorithm
        
    Returns:
        List of ClusterResult objects
    """
    if not ngrams_with_embeddings:
        return []
    
    items = [item[0] for item in ngrams_with_embeddings]
    scores = [item[1] for item in ngrams_with_embeddings]
    embeddings = [item[2] for item in ngrams_with_embeddings]
    
    clusterer = EmbeddingClusterer(algorithm=algorithm, n_clusters=n_clusters)
    return clusterer.cluster_embeddings(items, scores, embeddings)

def visualize_clusters_2d(embeddings: List[List[float]], 
                         labels: np.ndarray,
                         items: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create 2D visualization data for clusters using PCA
    
    Args:
        embeddings: List of embedding vectors
        labels: Cluster labels
        items: Optional list of item names
        
    Returns:
        Dictionary with visualization data
    """
    if not embeddings:
        return {}
    
    try:
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(np.array(embeddings))
        
        # Prepare visualization data
        viz_data = {
            'points': embeddings_2d.tolist(),
            'labels': labels.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'n_clusters': len(set(labels))
        }
        
        if items:
            viz_data['items'] = items
        
        return viz_data
        
    except Exception as e:
        logger.error(f"Failed to create visualization data: {e}")
        return {}

# Example usage and testing
if __name__ == "__main__":
    # Test data - create some sample embeddings
    np.random.seed(42)
    
    sample_items = [
        "python developer", "java developer", "javascript developer",
        "machine learning", "deep learning", "artificial intelligence", 
        "data science", "data analyst", "data engineer",
        "cloud computing", "aws engineer", "devops engineer",
        "frontend developer", "backend developer", "fullstack developer"
    ]
    
    sample_scores = np.random.uniform(1, 10, len(sample_items)).tolist()
    
    # Create sample embeddings (simulated)
    sample_embeddings = []
    for i, item in enumerate(sample_items):
        # Create somewhat realistic embeddings with patterns
        if "python" in item or "java" in item or "javascript" in item:
            base = np.array([1.0, 0.5, 0.2, 0.8, 0.3])
        elif "machine" in item or "deep" in item or "data" in item:
            base = np.array([0.3, 1.0, 0.7, 0.4, 0.9])
        elif "cloud" in item or "aws" in item or "devops" in item:
            base = np.array([0.6, 0.3, 1.0, 0.5, 0.4])
        else:
            base = np.array([0.5, 0.6, 0.4, 1.0, 0.7])
        
        # Add some noise
        noise = np.random.normal(0, 0.1, 5)
        embedding = (base + noise).tolist()
        sample_embeddings.append(embedding)
    
    print("🧪 Testing Clustering Module")
    print("=" * 50)
    
    # Test KMeans clustering
    clusterer = EmbeddingClusterer(algorithm='kmeans', n_clusters=5)
    cluster_results = clusterer.cluster_embeddings(sample_items, sample_scores, sample_embeddings)
    
    print(f"✅ Created {len(cluster_results)} clusters")
    
    for i, cluster in enumerate(cluster_results):
        print(f"\nCluster {cluster.cluster_id} (size: {cluster.size}, avg_score: {cluster.avg_score:.2f}):")
        for item in cluster.items[:5]:  # Show top 5 items
            print(f"  - {item}")
    
    print("\n" + "=" * 30)
    
    # Test optimal cluster finding
    if len(sample_embeddings) >= 10:
        optimal_results = clusterer.find_optimal_clusters(sample_embeddings, min_clusters=2, max_clusters=8)
        print("📊 Optimal cluster analysis:")
        for n_clusters, metrics in optimal_results.items():
            print(f"  {n_clusters} clusters - Silhouette: {metrics['silhouette_score']:.3f}")
    
    print("\n" + "=" * 30)
    
    # Test 2D visualization data
    labels = clusterer.fit_predict(sample_embeddings)
    viz_data = visualize_clusters_2d(sample_embeddings, labels, sample_items)
    
    if viz_data:
        print(f"✅ 2D visualization data created")
        print(f"   Explained variance: {viz_data['explained_variance_ratio']}")
        print(f"   Number of clusters: {viz_data['n_clusters']}")
