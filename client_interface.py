"""
Client Interface for Decentralized AI Compute Network
Submit jobs, check status, and retrieve results
"""

import pickle
import hashlib
import time
from typing import Dict, Any, Optional, List
import numpy as np

class NetworkClient:
    def __init__(self, client_address: str, contract):
        self.client_address = client_address
        self.contract = contract
        self.submitted_jobs: List[str] = []
        
    def get_balance(self) -> float:
        """Get token balance"""
        return self.contract.get_balance(self.client_address)
    
    def submit_linear_regression_job(self, X: np.ndarray, y: np.ndarray, 
                                    reward: float) -> str:
        """Submit a linear regression job"""
        # Combine X and y for transmission
        data = np.c_[X, y]
        
        job_data = {
            "model_type": "linear_regression",
            "input_data": data.tolist(),
            "params": {}
        }
        
        # Serialize for hashing
        input_bytes = pickle.dumps(data)
        
        # Submit to contract
        job_id = self.contract.submit_job(
            client_address=self.client_address,
            model_type="linear_regression",
            input_data=input_bytes,
            reward_tokens=reward
        )
        
        self.submitted_jobs.append(job_id)
        return job_id
    
    def submit_clustering_job(self, data: np.ndarray, k: int, 
                             reward: float) -> str:
        """Submit a K-means clustering job"""
        job_data = {
            "model_type": "kmeans_clustering",
            "input_data": data.tolist(),
            "params": {"k": k}
        }
        
        input_bytes = pickle.dumps(data)
        
        job_id = self.contract.submit_job(
            client_address=self.client_address,
            model_type="kmeans_clustering",
            input_data=input_bytes,
            reward_tokens=reward
        )
        
        self.submitted_jobs.append(job_id)
        return job_id
    
    def submit_pca_job(self, data: np.ndarray, n_components: int, 
                      reward: float) -> str:
        """Submit a PCA analysis job"""
        job_data = {
            "model_type": "pca_analysis",
            "input_data": data.tolist(),
            "params": {"n_components": n_components}
        }
        
        input_bytes = pickle.dumps(data)
        
        job_id = self.contract.submit_job(
            client_address=self.client_address,
            model_type="pca_analysis",
            input_data=input_bytes,
            reward_tokens=reward
        )
        
        self.submitted_jobs.append(job_id)
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a submitted job"""
        if job_id not in self.contract.jobs:
            return {"error": "Job not found"}
        
        job = self.contract.jobs[job_id]
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "worker": job.worker_address,
            "reward": job.reward_tokens,
            "submitted_at": job.timestamp,
            "completion_time": job.completion_time
        }
    
    def get_all_jobs_status(self) -> List[Dict[str, Any]]:
        """Get status of all submitted jobs"""
        return [self.get_job_status(jid) for jid in self.submitted_jobs]
    
    def verify_result(self, job_id: str, result_data: bytes, 
                     expected_hash: Optional[str] = None) -> bool:
        """Verify computation result"""
        if job_id not in self.contract.jobs:
            raise ValueError("Job not found")
        
        job = self.contract.jobs[job_id]
        
        # Verify result hash matches
        result_hash = hashlib.sha256(result_data).hexdigest()
        
        if job.result_hash != result_hash:
            return False
        
        # Additional verification logic can be added here
        # For proof-of-concept, we'll do basic checks
        
        try:
            result = pickle.loads(result_data)
            
            # Check result has expected structure
            if not isinstance(result, dict):
                return False
            
            if "model" not in result:
                return False
            
            # Model-specific validation
            if result["model"] == "linear_regression":
                if "coefficients" not in result or "mse" not in result:
                    return False
            elif result["model"] == "kmeans":
                if "centroids" not in result or "labels" not in result:
                    return False
            elif result["model"] == "pca":
                if "components" not in result or "explained_variance" not in result:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def finalize_job(self, job_id: str, is_valid: bool):
        """Verify and finalize a job (trigger payment)"""
        self.contract.verify_and_pay(job_id, is_valid)
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        jobs_by_status = {}
        for job_id in self.submitted_jobs:
            job = self.contract.jobs[job_id]
            status = job.status.value
            jobs_by_status[status] = jobs_by_status.get(status, 0) + 1
        
        return {
            "address": self.client_address,
            "balance": self.get_balance(),
            "total_jobs_submitted": len(self.submitted_jobs),
            "jobs_by_status": jobs_by_status
        }