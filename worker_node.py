"""
Worker Node for Decentralized AI Compute Network
Executes ML inference tasks and submits results
"""

import numpy as np
import hashlib
import pickle
import time
from typing import Dict, Any, Optional
import json

class MLModelRegistry:
    """Simple registry of supported ML models"""
    
    @staticmethod
    def linear_regression(data: np.ndarray) -> Dict[str, Any]:
        """Simple linear regression"""
        X = data[:, :-1]
        y = data[:, -1]
        
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Normal equation: theta = (X^T X)^-1 X^T y
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        # Make predictions
        predictions = X_b @ theta
        mse = np.mean((predictions - y) ** 2)
        
        return {
            "model": "linear_regression",
            "coefficients": theta.tolist(),
            "mse": float(mse),
            "predictions": predictions.tolist()
        }
    
    @staticmethod
    def kmeans_clustering(data: np.ndarray, k: int = 3) -> Dict[str, Any]:
        """K-means clustering"""
        np.random.seed(42)
        n_samples = data.shape[0]
        
        # Initialize centroids randomly
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = data[indices]
        
        # Run k-means for 10 iterations
        for _ in range(10):
            # Assign points to nearest centroid
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            for i in range(k):
                if np.sum(labels == i) > 0:
                    centroids[i] = data[labels == i].mean(axis=0)
        
        return {
            "model": "kmeans",
            "k": k,
            "centroids": centroids.tolist(),
            "labels": labels.tolist(),
            "inertia": float(np.sum((data - centroids[labels])**2))
        }
    
    @staticmethod
    def pca_analysis(data: np.ndarray, n_components: int = 2) -> Dict[str, Any]:
        """Principal Component Analysis"""
        # Center the data
        mean = np.mean(data, axis=0)
        centered = data - mean
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort by eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Project data
        components = eigenvectors[:, :n_components]
        transformed = centered @ components
        
        return {
            "model": "pca",
            "n_components": n_components,
            "explained_variance": eigenvalues[:n_components].tolist(),
            "components": components.tolist(),
            "transformed_data": transformed.tolist()
        }

class WorkerNode:
    def __init__(self, worker_id: str, gpu_type: str = "GTX 3090", 
                 compute_capacity: float = 35.5):
        self.worker_id = worker_id
        self.gpu_type = gpu_type
        self.compute_capacity = compute_capacity
        self.model_registry = MLModelRegistry()
        self.current_job: Optional[str] = None
        self.completed_jobs = []
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Return worker capabilities"""
        return {
            "worker_id": self.worker_id,
            "gpu_type": self.gpu_type,
            "compute_capacity": self.compute_capacity,
            "supported_models": [
                "linear_regression",
                "kmeans_clustering", 
                "pca_analysis"
            ],
            "status": "idle" if self.current_job is None else "busy"
        }
    
    def execute_job(self, job_data: Dict[str, Any]) -> bytes:
        """Execute a computation job"""
        model_type = job_data.get("model_type")
        input_data = job_data.get("input_data")
        params = job_data.get("params", {})
        
        # Deserialize input data
        data_array = np.array(input_data)
        
        # Route to appropriate model
        if model_type == "linear_regression":
            result = self.model_registry.linear_regression(data_array)
        elif model_type == "kmeans_clustering":
            k = params.get("k", 3)
            result = self.model_registry.kmeans_clustering(data_array, k)
        elif model_type == "pca_analysis":
            n_components = params.get("n_components", 2)
            result = self.model_registry.pca_analysis(data_array, n_components)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Add metadata
        result["worker_id"] = self.worker_id
        result["execution_time"] = time.time()
        result["gpu_type"] = self.gpu_type
        
        # Serialize result
        return pickle.dumps(result)
    
    def compute_work_proof(self, job_id: str, input_hash: str, 
                          result_data: bytes) -> str:
        """Generate proof of work"""
        proof_string = f"{job_id}{input_hash}{hashlib.sha256(result_data).hexdigest()}"
        return hashlib.sha256(proof_string.encode()).hexdigest()
    
    def process_job(self, job_id: str, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main job processing pipeline"""
        self.current_job = job_id
        
        start_time = time.time()
        
        try:
            # Execute computation
            result_data = self.execute_job(job_data)
            
            # Generate proof
            input_bytes = pickle.dumps(job_data.get("input_data"))
            input_hash = hashlib.sha256(input_bytes).hexdigest()
            proof = self.compute_work_proof(job_id, input_hash, result_data)
            
            execution_time = time.time() - start_time
            
            self.completed_jobs.append(job_id)
            self.current_job = None
            
            return {
                "job_id": job_id,
                "result_data": result_data,
                "proof": proof,
                "execution_time": execution_time,
                "success": True
            }
            
        except Exception as e:
            self.current_job = None
            return {
                "job_id": job_id,
                "error": str(e),
                "success": False
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        return {
            "worker_id": self.worker_id,
            "total_jobs_completed": len(self.completed_jobs),
            "current_job": self.current_job,
            "gpu_type": self.gpu_type,
            "compute_capacity": self.compute_capacity
        }