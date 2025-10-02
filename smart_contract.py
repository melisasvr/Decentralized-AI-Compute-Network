"""
Simulated Smart Contract for Decentralized AI Compute Network
Manages job queue, payments, and work verification
"""

import hashlib
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json

class JobStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPUTING = "computing"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FAILED = "failed"

@dataclass
class Job:
    job_id: str
    client_address: str
    model_type: str
    input_data_hash: str
    reward_tokens: float
    status: JobStatus
    worker_address: Optional[str] = None
    result_hash: Optional[str] = None
    timestamp: float = None
    completion_time: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class Worker:
    address: str
    gpu_type: str
    compute_capacity: float  # TFLOPS
    reputation_score: float
    total_jobs_completed: int
    stake_amount: float
    is_active: bool = True

class SmartContract:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.workers: Dict[str, Worker] = {}
        self.balances: Dict[str, float] = {}
        self.job_queue: List[str] = []
        self.contract_balance: float = 0
        self.min_stake = 100.0  # Minimum tokens to stake
        
    def register_worker(self, address: str, gpu_type: str, 
                       compute_capacity: float, stake_amount: float) -> bool:
        """Register a new worker node"""
        if stake_amount < self.min_stake:
            raise ValueError(f"Stake must be at least {self.min_stake} tokens")
        
        if address in self.workers:
            raise ValueError("Worker already registered")
        
        # Deduct stake from worker's balance
        if self.balances.get(address, 0) < stake_amount:
            raise ValueError("Insufficient balance for stake")
        
        self.balances[address] -= stake_amount
        self.contract_balance += stake_amount
        
        worker = Worker(
            address=address,
            gpu_type=gpu_type,
            compute_capacity=compute_capacity,
            reputation_score=100.0,
            total_jobs_completed=0,
            stake_amount=stake_amount
        )
        self.workers[address] = worker
        return True
    
    def submit_job(self, client_address: str, model_type: str,
                   input_data: bytes, reward_tokens: float) -> str:
        """Submit a new job to the network"""
        # Create job ID
        job_id = hashlib.sha256(
            f"{client_address}{time.time()}{model_type}".encode()
        ).hexdigest()[:16]
        
        # Calculate input hash for verification
        input_hash = hashlib.sha256(input_data).hexdigest()
        
        # Lock reward tokens
        if self.balances.get(client_address, 0) < reward_tokens:
            raise ValueError("Insufficient balance for reward")
        
        self.balances[client_address] -= reward_tokens
        self.contract_balance += reward_tokens
        
        # Create job
        job = Job(
            job_id=job_id,
            client_address=client_address,
            model_type=model_type,
            input_data_hash=input_hash,
            reward_tokens=reward_tokens,
            status=JobStatus.PENDING
        )
        
        self.jobs[job_id] = job
        self.job_queue.append(job_id)
        
        return job_id
    
    def assign_job(self, job_id: str, worker_address: str) -> bool:
        """Assign a job to a worker"""
        if job_id not in self.jobs:
            raise ValueError("Job not found")
        
        if worker_address not in self.workers:
            raise ValueError("Worker not registered")
        
        job = self.jobs[job_id]
        worker = self.workers[worker_address]
        
        if job.status != JobStatus.PENDING:
            raise ValueError("Job not available for assignment")
        
        if not worker.is_active:
            raise ValueError("Worker is not active")
        
        job.worker_address = worker_address
        job.status = JobStatus.ASSIGNED
        
        if job_id in self.job_queue:
            self.job_queue.remove(job_id)
        
        return True
    
    def submit_result(self, job_id: str, worker_address: str, 
                     result_data: bytes) -> bool:
        """Worker submits computation result"""
        if job_id not in self.jobs:
            raise ValueError("Job not found")
        
        job = self.jobs[job_id]
        
        if job.worker_address != worker_address:
            raise ValueError("Unauthorized worker")
        
        if job.status not in [JobStatus.ASSIGNED, JobStatus.COMPUTING]:
            raise ValueError("Invalid job status")
        
        # Calculate result hash
        result_hash = hashlib.sha256(result_data).hexdigest()
        job.result_hash = result_hash
        job.status = JobStatus.COMPLETED
        job.completion_time = time.time()
        
        return True
    
    def verify_and_pay(self, job_id: str, is_valid: bool) -> bool:
        """Verify work and distribute payment"""
        if job_id not in self.jobs:
            raise ValueError("Job not found")
        
        job = self.jobs[job_id]
        
        if job.status != JobStatus.COMPLETED:
            raise ValueError("Job not ready for verification")
        
        worker = self.workers[job.worker_address]
        
        if is_valid:
            # Pay worker
            self.balances[job.worker_address] = \
                self.balances.get(job.worker_address, 0) + job.reward_tokens
            self.contract_balance -= job.reward_tokens
            
            # Update worker stats
            worker.total_jobs_completed += 1
            worker.reputation_score = min(100.0, worker.reputation_score + 0.5)
            
            job.status = JobStatus.VERIFIED
        else:
            # Slash worker stake and return to client
            slash_amount = min(worker.stake_amount * 0.1, job.reward_tokens)
            worker.stake_amount -= slash_amount
            
            self.balances[job.client_address] = \
                self.balances.get(job.client_address, 0) + job.reward_tokens
            self.contract_balance -= job.reward_tokens
            
            worker.reputation_score = max(0, worker.reputation_score - 10)
            
            job.status = JobStatus.FAILED
            
            # Deactivate if reputation too low
            if worker.reputation_score < 50:
                worker.is_active = False
        
        return True
    
    def get_available_jobs(self) -> List[Job]:
        """Get list of pending jobs"""
        return [self.jobs[jid] for jid in self.job_queue 
                if self.jobs[jid].status == JobStatus.PENDING]
    
    def get_worker_jobs(self, worker_address: str) -> List[Job]:
        """Get jobs assigned to a worker"""
        return [job for job in self.jobs.values() 
                if job.worker_address == worker_address]
    
    def get_balance(self, address: str) -> float:
        """Get token balance for an address"""
        return self.balances.get(address, 0)
    
    def mint_tokens(self, address: str, amount: float):
        """Mint tokens (for testing purposes)"""
        self.balances[address] = self.balances.get(address, 0) + amount
    
    def get_contract_state(self) -> Dict:
        """Get current contract state"""
        return {
            "total_jobs": len(self.jobs),
            "pending_jobs": len(self.job_queue),
            "total_workers": len(self.workers),
            "active_workers": sum(1 for w in self.workers.values() if w.is_active),
            "contract_balance": self.contract_balance
        }
    
    def export_state(self) -> str:
        """Export contract state to JSON"""
        state = {
            "jobs": {k: asdict(v) for k, v in self.jobs.items()},
            "workers": {k: asdict(v) for k, v in self.workers.items()},
            "balances": self.balances,
            "job_queue": self.job_queue,
            "contract_balance": self.contract_balance
        }
        return json.dumps(state, indent=2, default=str)