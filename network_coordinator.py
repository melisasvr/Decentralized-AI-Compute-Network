"""
Network Coordinator for Decentralized AI Compute Network
Manages job assignment, worker selection, and result verification
"""

from typing import Dict, List, Optional
import pickle
import time

class NetworkCoordinator:
    def __init__(self, contract):
        self.contract = contract
        self.worker_nodes: Dict[str, object] = {}
        self.result_cache: Dict[str, bytes] = {}
        
    def register_worker_node(self, worker_node):
        """Register a worker node with the coordinator"""
        self.worker_nodes[worker_node.worker_id] = worker_node
        
    def get_available_workers(self) -> List[Dict]:
        """Get list of available workers"""
        available = []
        for worker_id, worker in self.worker_nodes.items():
            if worker_id in self.contract.workers:
                contract_worker = self.contract.workers[worker_id]
                if contract_worker.is_active and worker.current_job is None:
                    available.append({
                        "worker_id": worker_id,
                        "gpu_type": worker.gpu_type,
                        "compute_capacity": worker.compute_capacity,
                        "reputation": contract_worker.reputation_score,
                        "completed_jobs": contract_worker.total_jobs_completed
                    })
        return available
    
    def select_worker_for_job(self, job_id: str) -> Optional[str]:
        """Select best worker for a job based on reputation and availability"""
        available = self.get_available_workers()
        
        if not available:
            return None
        
        # Sort by reputation score (descending)
        available.sort(key=lambda w: w["reputation"], reverse=True)
        
        # Return best worker
        return available[0]["worker_id"]
    
    def assign_jobs(self) -> List[str]:
        """Assign pending jobs to available workers"""
        assigned = []
        pending_jobs = self.contract.get_available_jobs()
        
        for job in pending_jobs:
            worker_id = self.select_worker_for_job(job.job_id)
            
            if worker_id:
                try:
                    self.contract.assign_job(job.job_id, worker_id)
                    assigned.append(job.job_id)
                    print(f"Assigned job {job.job_id} to worker {worker_id}")
                except Exception as e:
                    print(f"Failed to assign job {job.job_id}: {e}")
        
        return assigned
    
    def execute_assigned_jobs(self) -> Dict[str, bool]:
        """Execute jobs assigned to workers"""
        results = {}
        
        for worker_id, worker in self.worker_nodes.items():
            # Get jobs assigned to this worker
            worker_jobs = self.contract.get_worker_jobs(worker_id)
            
            for job in worker_jobs:
                if job.status.value == "assigned":
                    # Mark as computing
                    job.status = self.contract.jobs[job.job_id].status
                    
                    # Prepare job data
                    job_data = self._prepare_job_data(job)
                    
                    # Execute job
                    result = worker.process_job(job.job_id, job_data)
                    
                    if result["success"]:
                        # Submit result to contract
                        self.contract.submit_result(
                            job.job_id,
                            worker_id,
                            result["result_data"]
                        )
                        
                        # Cache result
                        self.result_cache[job.job_id] = result["result_data"]
                        
                        results[job.job_id] = True
                        print(f"Job {job.job_id} completed by {worker_id}")
                    else:
                        results[job.job_id] = False
                        print(f"Job {job.job_id} failed: {result.get('error')}")
        
        return results
    
    def _prepare_job_data(self, job) -> Dict:
        """Prepare job data for execution"""
        # In a real system, this would fetch data from distributed storage
        # For POC, we'll create sample data based on model type
        
        if job.model_type == "linear_regression":
            # Generate sample data
            import numpy as np
            X = np.random.randn(100, 3)
            y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1
            data = np.c_[X, y]
            
            return {
                "model_type": job.model_type,
                "input_data": data.tolist(),
                "params": {}
            }
        
        elif job.model_type == "kmeans_clustering":
            import numpy as np
            data = np.random.randn(200, 2)
            
            return {
                "model_type": job.model_type,
                "input_data": data.tolist(),
                "params": {"k": 3}
            }
        
        elif job.model_type == "pca_analysis":
            import numpy as np
            data = np.random.randn(150, 5)
            
            return {
                "model_type": job.model_type,
                "input_data": data.tolist(),
                "params": {"n_components": 2}
            }
        
        return {}
    
    def verify_and_finalize_jobs(self):
        """Verify completed jobs and trigger payments"""
        for job_id, job in self.contract.jobs.items():
            if job.status.value == "completed":
                if job_id in self.result_cache:
                    result_data = self.result_cache[job_id]
                    
                    # Basic verification - in real system would be more sophisticated
                    try:
                        result = pickle.loads(result_data)
                        is_valid = isinstance(result, dict) and "model" in result
                    except:
                        is_valid = False
                    
                    # Finalize job
                    self.contract.verify_and_pay(job_id, is_valid)
                    print(f"Job {job_id} verified: {is_valid}")
    
    def run_coordination_cycle(self):
        """Run one cycle of coordination"""
        print("\n=== Running Coordination Cycle ===")
        
        # Step 1: Assign pending jobs
        assigned = self.assign_jobs()
        print(f"Assigned {len(assigned)} jobs")
        
        # Step 2: Execute assigned jobs
        executed = self.execute_assigned_jobs()
        print(f"Executed {len(executed)} jobs")
        
        # Step 3: Verify and finalize completed jobs
        self.verify_and_finalize_jobs()
        
        # Step 4: Display network stats
        self.print_network_stats()
    
    def print_network_stats(self):
        """Print current network statistics"""
        state = self.contract.get_contract_state()
        print(f"\n--- Network Statistics ---")
        print(f"Total Workers: {state['total_workers']}")
        print(f"Active Workers: {state['active_workers']}")
        print(f"Total Jobs: {state['total_jobs']}")
        print(f"Pending Jobs: {state['pending_jobs']}")
        print(f"Contract Balance: {state['contract_balance']:.2f} tokens")
    
    def get_job_result(self, job_id: str) -> Optional[Dict]:
        """Retrieve result for a completed job"""
        if job_id not in self.result_cache:
            return None
        
        try:
            result = pickle.loads(self.result_cache[job_id])
            return result
        except:
            return None
    
    def monitor_workers(self) -> Dict[str, Dict]:
        """Monitor health and status of all workers"""
        worker_status = {}
        
        for worker_id, worker_node in self.worker_nodes.items():
            if worker_id in self.contract.workers:
                contract_worker = self.contract.workers[worker_id]
                
                worker_status[worker_id] = {
                    "is_active": contract_worker.is_active,
                    "reputation": contract_worker.reputation_score,
                    "completed_jobs": contract_worker.total_jobs_completed,
                    "stake": contract_worker.stake_amount,
                    "current_job": worker_node.current_job,
                    "gpu_type": worker_node.gpu_type
                }
        
        return worker_status
    
    def get_network_performance(self) -> Dict:
        """Get overall network performance metrics"""
        all_jobs = list(self.contract.jobs.values())
        
        if not all_jobs:
            return {
                "total_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "average_completion_time": 0,
                "success_rate": 0
            }
        
        completed = [j for j in all_jobs if j.status.value == "verified"]
        failed = [j for j in all_jobs if j.status.value == "failed"]
        
        completion_times = [
            j.completion_time - j.timestamp 
            for j in completed 
            if j.completion_time
        ]
        
        avg_time = sum(completion_times) / len(completion_times) if completion_times else 0
        success_rate = len(completed) / len(all_jobs) * 100 if all_jobs else 0
        
        return {
            "total_jobs": len(all_jobs),
            "completed_jobs": len(completed),
            "failed_jobs": len(failed),
            "average_completion_time": avg_time,
            "success_rate": success_rate
        }