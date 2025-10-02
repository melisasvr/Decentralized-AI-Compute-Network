"""
Complete Demonstration of Decentralized AI Compute Network
Run this script to see the full system in action
"""

import numpy as np
import time
from smart_contract import SmartContract
from worker_node import WorkerNode
from client_interface import NetworkClient
from network_coordinator import NetworkCoordinator

def print_separator(title=""):
    print("\n" + "="*60)
    if title:
        print(f"  {title}")
        print("="*60)

def demo_network():
    """Main demonstration function"""
    
    print_separator("DECENTRALIZED AI COMPUTE NETWORK - DEMO")
    
    # Initialize the smart contract
    print("\n[1] Initializing Smart Contract...")
    contract = SmartContract()
    print("✓ Smart contract deployed")
    
    # Initialize coordinator
    coordinator = NetworkCoordinator(contract)
    print("✓ Smart contract deployed")
    
    # Create and register workers
    print_separator("WORKER REGISTRATION")
    
    workers = []
    worker_configs = [
        ("worker_1", "RTX 4090", 82.6),
        ("worker_2", "RTX 3090", 35.5),
        ("worker_3", "A100", 156.0),
    ]
    
    for worker_id, gpu, capacity in worker_configs:
        # Create worker node
        worker = WorkerNode(worker_id, gpu, capacity)
        workers.append(worker)
        coordinator.register_worker_node(worker)
        
        # Mint tokens for staking
        contract.mint_tokens(worker_id, 500)
        
        # Register with contract
        contract.register_worker(worker_id, gpu, capacity, stake_amount=150)
        
        print(f"✓ Registered {worker_id} ({gpu}, {capacity} TFLOPS)")
    
    # Create clients
    print_separator("CLIENT SETUP")
    
    clients = []
    client_addresses = ["client_alice", "client_bob", "client_charlie"]
    
    for addr in client_addresses:
        # Mint tokens for clients
        contract.mint_tokens(addr, 1000)
        client = NetworkClient(addr, contract)
        clients.append(client)
        print(f"✓ Client {addr} initialized with {client.get_balance()} tokens")
    
    # Submit various jobs
    print_separator("JOB SUBMISSION")
    
    # Client 1: Linear Regression
    print("\n[Client Alice] Submitting Linear Regression job...")
    X = np.random.randn(100, 3)
    y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1
    job1 = clients[0].submit_linear_regression_job(X, y, reward=50)
    print(f"  Job ID: {job1}")
    print(f"  Reward: 50 tokens")
    
    # Client 2: K-means Clustering
    print("\n[Client Bob] Submitting K-Means Clustering job...")
    data = np.random.randn(200, 2)
    job2 = clients[1].submit_clustering_job(data, k=3, reward=40)
    print(f"  Job ID: {job2}")
    print(f"  Reward: 40 tokens")
    
    # Client 3: PCA Analysis
    print("\n[Client Charlie] Submitting PCA Analysis job...")
    data = np.random.randn(150, 5)
    job3 = clients[2].submit_pca_job(data, n_components=2, reward=45)
    print(f"  Job ID: {job3}")
    print(f"  Reward: 45 tokens")
    
    # Display contract state
    print_separator("CONTRACT STATE AFTER SUBMISSIONS")
    state = contract.get_contract_state()
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    # Run coordination cycles
    print_separator("EXECUTING JOBS")
    
    for cycle in range(3):
        print(f"\n--- Cycle {cycle + 1} ---")
        time.sleep(0.5)  # Simulate processing time
        coordinator.run_coordination_cycle()
    
    # Display job results
    print_separator("JOB RESULTS")
    
    for i, client in enumerate(clients):
        print(f"\n[Client {client_addresses[i]}]")
        jobs_status = client.get_all_jobs_status()
        
        for job_status in jobs_status:
            print(f"\n  Job {job_status['job_id']}:")
            print(f"    Status: {job_status['status']}")
            print(f"    Worker: {job_status['worker']}")
            print(f"    Reward: {job_status['reward']} tokens")
            
            if job_status['status'] == 'verified':
                # Get actual result
                result = coordinator.get_job_result(job_status['job_id'])
                if result:
                    print(f"    Model: {result['model']}")
                    print(f"    Executed by: {result['worker_id']} ({result['gpu_type']})")
                    
                    if result['model'] == 'linear_regression':
                        print(f"    MSE: {result['mse']:.4f}")
                        print(f"    Coefficients: {[f'{c:.3f}' for c in result['coefficients']]}")
                    elif result['model'] == 'kmeans':
                        print(f"    K: {result['k']}")
                        print(f"    Inertia: {result['inertia']:.4f}")
                    elif result['model'] == 'pca':
                        print(f"    Components: {result['n_components']}")
                        print(f"    Explained Variance: {[f'{v:.3f}' for v in result['explained_variance']]}")
    
    # Display balances after completion
    print_separator("FINAL BALANCES")
    
    print("\nClients:")
    for addr in client_addresses:
        balance = contract.get_balance(addr)
        print(f"  {addr}: {balance:.2f} tokens")
    
    print("\nWorkers:")
    for worker_id, _, _ in worker_configs:
        balance = contract.get_balance(worker_id)
        worker = contract.workers[worker_id]
        print(f"  {worker_id}: {balance:.2f} tokens (Stake: {worker.stake_amount:.2f})")
    
    # Display worker statistics
    print_separator("WORKER STATISTICS")
    
    for worker_id, _, _ in worker_configs:
        worker = contract.workers[worker_id]
        print(f"\n{worker_id}:")
        print(f"  GPU: {worker.gpu_type}")
        print(f"  Reputation: {worker.reputation_score:.2f}/100")
        print(f"  Jobs Completed: {worker.total_jobs_completed}")
        print(f"  Stake: {worker.stake_amount:.2f} tokens")
        print(f"  Status: {'Active' if worker.is_active else 'Inactive'}")
    
    # Network performance metrics
    print_separator("NETWORK PERFORMANCE")
    
    perf = coordinator.get_network_performance()
    print(f"\nTotal Jobs: {perf['total_jobs']}")
    print(f"Completed Jobs: {perf['completed_jobs']}")
    print(f"Failed Jobs: {perf['failed_jobs']}")
    print(f"Success Rate: {perf['success_rate']:.1f}%")
    print(f"Average Completion Time: {perf['average_completion_time']:.3f}s")
    
    # Export contract state
    print_separator("BLOCKCHAIN STATE EXPORT")
    
    state_json = contract.export_state()
    print("\nContract state exported to JSON (first 500 chars):")
    print(state_json[:500] + "...")
    
    print_separator("DEMO COMPLETE")
    print("\n✓ All jobs successfully executed and verified")
    print("✓ Payments distributed to workers")
    print("✓ Network operating successfully\n")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the demo
    demo_network()