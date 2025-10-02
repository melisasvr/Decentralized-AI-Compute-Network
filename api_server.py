"""
REST API Server for Decentralized AI Compute Network
Provides HTTP endpoints for clients and workers

Install dependencies:
    pip install flask flask-cors numpy

Run server:
    python api_server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import base64
from smart_contract import SmartContract
from worker_node import WorkerNode
from client_interface import NetworkClient
from network_coordinator import NetworkCoordinator
import threading
import time

app = Flask(__name__)
CORS(app)

# Global state
contract = SmartContract()
coordinator = NetworkCoordinator(contract)
worker_nodes = {}

# Background thread for coordination
coordination_active = False

def coordination_loop():
    """Background thread to run coordination cycles"""
    while coordination_active:
        try:
            coordinator.run_coordination_cycle()
        except Exception as e:
            print(f"Coordination error: {e}")
        time.sleep(5)  # Run every 5 seconds

# ============= Worker Endpoints =============

@app.route('/worker/register', methods=['POST'])
def register_worker():
    """Register a new worker node"""
    try:
        data = request.json
        worker_id = data['worker_id']
        gpu_type = data['gpu_type']
        compute_capacity = data['compute_capacity']
        stake_amount = data['stake_amount']
        
        # Mint tokens for worker (in production, they'd need to acquire tokens)
        contract.mint_tokens(worker_id, stake_amount + 100)
        
        # Register with contract
        contract.register_worker(worker_id, gpu_type, compute_capacity, stake_amount)
        
        # Create worker node
        worker = WorkerNode(worker_id, gpu_type, compute_capacity)
        worker_nodes[worker_id] = worker
        coordinator.register_worker_node(worker)
        
        return jsonify({
            "success": True,
            "worker_id": worker_id,
            "balance": contract.get_balance(worker_id)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/worker/<worker_id>/status', methods=['GET'])
def get_worker_status(worker_id):
    """Get worker status and statistics"""
    try:
        if worker_id not in contract.workers:
            return jsonify({"error": "Worker not found"}), 404
        
        worker = contract.workers[worker_id]
        worker_node = worker_nodes.get(worker_id)
        
        return jsonify({
            "worker_id": worker_id,
            "gpu_type": worker.gpu_type,
            "compute_capacity": worker.compute_capacity,
            "reputation": worker.reputation_score,
            "jobs_completed": worker.total_jobs_completed,
            "stake": worker.stake_amount,
            "is_active": worker.is_active,
            "current_job": worker_node.current_job if worker_node else None,
            "balance": contract.get_balance(worker_id)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/workers', methods=['GET'])
def list_workers():
    """List all registered workers"""
    workers = []
    for worker_id, worker in contract.workers.items():
        workers.append({
            "worker_id": worker_id,
            "gpu_type": worker.gpu_type,
            "reputation": worker.reputation_score,
            "jobs_completed": worker.total_jobs_completed,
            "is_active": worker.is_active
        })
    
    return jsonify({"workers": workers, "count": len(workers)})

# ============= Client Endpoints =============

@app.route('/client/register', methods=['POST'])
def register_client():
    """Register a new client (mint tokens)"""
    try:
        data = request.json
        client_address = data['client_address']
        initial_balance = data.get('initial_balance', 1000)
        
        # Mint tokens
        contract.mint_tokens(client_address, initial_balance)
        
        return jsonify({
            "success": True,
            "client_address": client_address,
            "balance": contract.get_balance(client_address)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/client/<client_address>/balance', methods=['GET'])
def get_client_balance(client_address):
    """Get client token balance"""
    return jsonify({
        "client_address": client_address,
        "balance": contract.get_balance(client_address)
    })

@app.route('/clients', methods=['GET'])
def list_clients():
    """List all registered clients with their balances"""
    clients = []
    
    # Get all addresses that have tokens (excluding workers)
    for address, balance in contract.balances.items():
        # Check if this address is not a worker
        if address not in contract.workers:
            clients.append({
                "client_address": address,
                "balance": balance
            })
    
    return jsonify({"clients": clients, "count": len(clients)})

# ============= Job Endpoints =============

@app.route('/job/submit', methods=['POST'])
def submit_job():
    """Submit a new computation job"""
    try:
        data = request.json
        client_address = data['client_address']
        model_type = data['model_type']
        reward = data['reward']
        
        # Decode input data
        input_data_b64 = data.get('input_data')
        if input_data_b64:
            input_bytes = base64.b64decode(input_data_b64)
        else:
            # Generate sample data if none provided
            if model_type == "linear_regression":
                X = np.random.randn(100, 3)
                y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1
                input_bytes = pickle.dumps(np.c_[X, y])
            elif model_type == "kmeans_clustering":
                input_bytes = pickle.dumps(np.random.randn(200, 2))
            elif model_type == "pca_analysis":
                input_bytes = pickle.dumps(np.random.randn(150, 5))
            else:
                return jsonify({"error": "Invalid model type"}), 400
        
        # Submit job
        job_id = contract.submit_job(
            client_address=client_address,
            model_type=model_type,
            input_data=input_bytes,
            reward_tokens=reward
        )
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "pending"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/job/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    """Get job status"""
    try:
        if job_id not in contract.jobs:
            return jsonify({"error": "Job not found"}), 404
        
        job = contract.jobs[job_id]
        
        response = {
            "job_id": job.job_id,
            "client": job.client_address,
            "model_type": job.model_type,
            "status": job.status.value,
            "reward": job.reward_tokens,
            "worker": job.worker_address,
            "submitted_at": job.timestamp,
            "completion_time": job.completion_time
        }
        
        # Include result if completed
        if job.status.value in ["completed", "verified"]:
            result = coordinator.get_job_result(job_id)
            if result:
                response["result"] = result
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs with optional filtering"""
    status_filter = request.args.get('status')
    client_filter = request.args.get('client')
    
    jobs = []
    for job in contract.jobs.values():
        if status_filter and job.status.value != status_filter:
            continue
        if client_filter and job.client_address != client_filter:
            continue
        
        jobs.append({
            "job_id": job.job_id,
            "client": job.client_address,
            "model_type": job.model_type,
            "status": job.status.value,
            "reward": job.reward_tokens,
            "worker": job.worker_address
        })
    
    return jsonify({"jobs": jobs, "count": len(jobs)})

@app.route('/jobs/pending', methods=['GET'])
def get_pending_jobs():
    """Get list of pending jobs"""
    pending = contract.get_available_jobs()
    
    jobs = [{
        "job_id": job.job_id,
        "model_type": job.model_type,
        "reward": job.reward_tokens,
        "timestamp": job.timestamp
    } for job in pending]
    
    return jsonify({"jobs": jobs, "count": len(jobs)})

# ============= Network Endpoints =============

@app.route('/network/stats', methods=['GET'])
def get_network_stats():
    """Get network statistics"""
    state = contract.get_contract_state()
    performance = coordinator.get_network_performance()
    
    return jsonify({
        "contract_state": state,
        "performance": performance
    })

@app.route('/network/coordination/start', methods=['POST'])
def start_coordination():
    """Start background coordination"""
    global coordination_active
    
    if coordination_active:
        return jsonify({"message": "Coordination already running"})
    
    coordination_active = True
    thread = threading.Thread(target=coordination_loop, daemon=True)
    thread.start()
    
    return jsonify({"success": True, "message": "Coordination started"})

@app.route('/network/coordination/stop', methods=['POST'])
def stop_coordination():
    """Stop background coordination"""
    global coordination_active
    coordination_active = False
    
    return jsonify({"success": True, "message": "Coordination stopped"})

@app.route('/network/coordination/trigger', methods=['POST'])
def trigger_coordination():
    """Manually trigger a coordination cycle"""
    try:
        coordinator.run_coordination_cycle()
        return jsonify({"success": True, "message": "Coordination cycle completed"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ============= Health Check =============

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "coordination_active": coordination_active,
        "workers": len(worker_nodes),
        "jobs": len(contract.jobs)
    })

@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return jsonify({
        "name": "Decentralized AI Compute Network API",
        "version": "1.0.0",
        "endpoints": {
            "worker": {
                "POST /worker/register": "Register a new worker",
                "GET /worker/<id>/status": "Get worker status",
                "GET /workers": "List all workers"
            },
            "client": {
                "POST /client/register": "Register a client",
                "GET /client/<address>/balance": "Get client balance",
                "GET /clients": "List all clients"
            },
            "jobs": {
                "POST /job/submit": "Submit a new job",
                "GET /job/<id>/status": "Get job status",
                "GET /jobs": "List all jobs",
                "GET /jobs/pending": "Get pending jobs"
            },
            "network": {
                "GET /network/stats": "Get network statistics",
                "POST /network/coordination/start": "Start auto-coordination",
                "POST /network/coordination/stop": "Stop auto-coordination",
                "POST /network/coordination/trigger": "Trigger coordination cycle"
            }
        }
    })

if __name__ == '__main__':
    print("="*60)
    print("  Decentralized AI Compute Network API")
    print("="*60)
    print("\nStarting server on http://localhost:5000")
    print("\nAPI Documentation: http://localhost:5000/")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)