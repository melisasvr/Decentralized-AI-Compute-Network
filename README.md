# Decentralized AI Compute Network
- A proof-of-concept decentralized network where users can rent out their unused GPU computing power to run machine learning models.
- Similar to Render Network or Bittensor, this system uses smart contracts to manage job queues, token-based payments, and verification of computational work.

## 🌟Features 
- **Decentralized GPU Marketplace**: Workers register their GPUs and earn tokens by processing ML tasks
- **Smart Contract Management**: Automated job assignment, payment distribution, and result verification
- **Token Economy**: ERC20-like token system for payments and staking
- **Reputation System**: Workers build reputation through successful job completion
- **Multiple ML Models**: Support for Linear Regression, K-Means Clustering, and PCA Analysis
- **REST API**: Full HTTP API for programmatic access
- **Web Dashboard**: Beautiful, real-time dashboard for monitoring and interaction
- **Auto-Coordination**: Background processing of pending jobs

## 🏗️Architecture

```
┌─────────────────┐
│  Web Dashboard  │
└────────┬────────┘
         │
         │ HTTP
         │
┌────────▼────────┐
│   REST API      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────────┐
│Smart  │ │  Network    │
│Contract│─│ Coordinator │
└───┬───┘ └──┬──────────┘
    │        │
    │    ┌───▼────┐
    │    │ Worker │
    │    │ Nodes  │
    └────┴────────┘
```

## 🧩 Components
### 1. Smart Contract (`smart_contract.py`)
- Manages token balances and transactions
- Handles worker registration with staking
- Tracks job lifecycle (pending → assigned → completed → verified)
- Implements a reputation scoring system
- Provides contract state export for blockchain integration

### 2. Worker Node (`worker_node.py`)
- Processes machine learning jobs
- Executes Linear Regression, K-Means, and PCA algorithms
- Reports results back to the network
- Tracks GPU utilization

### 3. Network Coordinator (`network_coordinator.py`)
- Assigns jobs to workers based on reputation
- Monitors worker availability and health
- Verifies computation results
- Manages coordination cycles
- Caches results for retrieval

### 4. Client Interface (`client_interface.py`)
- Provides an API for job submission
- Tracks job status
- Manages client token balance
- Retrieves computation results

### 5. REST API Server (`api_server.py`)
- HTTP endpoints for all operations
- Background coordination thread
- CORS-enabled for web access

### 6. Web Dashboard (`dashboard.html`)
- Real-time network statistics
- Worker and client registration forms
- Job submission interface
- Live monitoring of workers, clients, and jobs
- Network control panel

## 🚀Installation

### Prerequisites

```bash
Python 3.8+
pip
```

### Install Dependencies

```bash
pip install flask flask-cors numpy scikit-learn
```

## 🏁 Quick Start

### 1. Run the Demo Script

See the entire system in action with a pre-configured demo:

```bash
python demo_script.py
```

This will:
- Initialize the smart contract
- Register 3 workers with different GPUs
- Register 3 clients with token balances
- Submit 3 different ML jobs
- Execute and verify all jobs
- Display complete results and statistics

### 2. Start the API Server

```bash
python api_server.py
```

The server will start on `http://localhost:5000`

### 3. Open the Web Dashboard
- Open `dashboard.html` in your browser, or navigate to your local file path.
- The dashboard connects to `http://localhost:5000` automatically.

## 💻 Usage Guide
### Registering a Worker

**Via Dashboard:**
1. Fill in Worker ID (e.g., `worker_rtx4090_01`)
2. Select GPU Type from the dropdown
3. Compute Capacity auto-fills
4. Enter Stake Amount (e.g., `150` tokens)
5. Click "Register Worker"

**Via API:**
```bash
curl -X POST http://localhost:5000/worker/register \
  -H "Content-Type: application/json" \
  -d '{
    "worker_id": "worker_rtx4090_01",
    "gpu_type": "RTX 4090",
    "compute_capacity": 82.6,
    "stake_amount": 150
  }'
```

### 🖥️Registering a Client 

**Via Dashboard:**
1. Enter Client Address (e.g., `alice`)
2. Enter Initial Balance (e.g., `1000` tokens)
3. Click "Register Client"

**Via API:**
```bash
curl -X POST http://localhost:5000/client/register \
  -H "Content-Type: application/json" \
  -d '{
    "client_address": "alice",
    "initial_balance": 1000
  }'
```

### 💼Submitting a Job 

**Via Dashboard:**
1. Enter your Client Address
2. Select Model Type (Linear Regression, K-Means, or PCA)
3. Enter the Reward amount in tokens
4. Click "Submit Job"

**Via API:**
```bash
curl -X POST http://localhost:5000/job/submit \
  -H "Content-Type: application/json" \
  -d '{
    "client_address": "alice",
    "model_type": "linear_regression",
    "reward": 50
  }'
```

### Processing Jobs  

**Auto-Coordination (Recommended):**
Click "Start Auto-Coordination" in the dashboard, or:
```bash
curl -X POST http://localhost:5000/network/coordination/start
```

**Manual Trigger:**
Click "Trigger Cycle" in the dashboard, or:
```bash
curl -X POST http://localhost:5000/network/coordination/trigger
```

## 🔑API Endpoints
### Worker Endpoints
- `POST /worker/register` - Register a new worker
- `GET /worker/<id>/status` - Get worker status
- `GET /workers` - List all workers

### Client Endpoints
- `POST /client/register` - Register a client
- `GET /client/<address>/balance` - Get client balance
- `GET /clients` - List all clients

### Job Endpoints
- `POST /job/submit` - Submit a new job
- `GET /job/<id>/status` - Get job status
- `GET /jobs` - List all jobs
- `GET /jobs/pending` - Get pending jobs

### Network Endpoints
- `GET /network/stats` - Get network statistics
- `POST /network/coordination/start` - Start auto-coordination
- `POST /network/coordination/stop` - Stop auto-coordination
- `POST /network/coordination/trigger` - Trigger coordination cycle

### Health Check
- `GET /health` - Health check endpoint
- `GET /` - API documentation

## 🦾Supported ML Models
### 1. Linear Regression
Fits a linear model to predict continuous values.

**Input:** Feature matrix and target vector  
**Output:** Coefficients, intercept, MSE

### 2. K-Means Clustering
Groups data points into K clusters.

**Input:** Data points and number of clusters  
**Output:** Cluster centers, labels, inertia

### 3. PCA Analysis
- Dimensionality reduction using Principal Component Analysis.

**Input:** High-dimensional data and number of components  
**Output:** Principal components, explained variance

## 🪙Token Economics
- **Worker Staking**: Workers must stake tokens to participate (prevents malicious behavior)
- **Job Rewards**: Clients pay tokens for computation
- **Payment Distribution**: Tokens transferred from client to worker upon job verification
- **Reputation Impact**: Failed jobs reduce reputation and may result in slashing

## ⚙️ System Flow
1. **Worker Registration**
   - Worker stakes tokens
   - GPU specs registered in smart contract
   - Initial reputation score: 100/100

2. **Client Registration**
   - Client receives initial token balance
   - Can submit jobs and track balance

3. **Job Submission**
   - Client submits a job with a reward
   - Tokens locked in contract
   - Job enters pending queue

4. **Job Assignment**
   - Coordinator selects best available worker (by reputation)
   - Job assigned to the worker
   - Status: pending → assigned

5. **Job Execution**
   - Worker processes ML task
   - Results computed and submitted
   - Status: assigned → completed

6. **Verification & Payment**
   - Coordinator verifies results
   - Tokens transferred to the worker
   - Reputation updated
   - Status: completed → verified

## 📊Dashboard Features

### 📈Statistics Cards
- Total Workers / Active Workers
- Total Jobs / Completed Jobs
- Success Rate
- Contract Balance

### 📝Registration Forms
- Worker registration with GPU selection
- Client registration with token allocation
- Job submission with model selection

### 🌐Network Control
- Start/Stop auto-coordination
- Manual coordination trigger
- Refresh data

### 🔍 Live Monitoring
- Active Workers panel (GPU, reputation, jobs completed)
- Registered Clients panel (address, token balance)
- Recent Jobs panel (status, worker assignment, rewards)

### 🔄Auto-Refresh
Dashboard refreshes every 5 seconds to show the latest data

## Testing Scenarios

### Basic Flow Test
```bash
# 1. Start API server
python api_server.py

# 2. Register a worker (in another terminal)
curl -X POST http://localhost:5000/worker/register \
  -H "Content-Type: application/json" \
  -d '{"worker_id": "test_worker", "gpu_type": "RTX 4090", "compute_capacity": 82.6, "stake_amount": 150}'

# 3. Register a client
curl -X POST http://localhost:5000/client/register \
  -H "Content-Type: application/json" \
  -d '{"client_address": "test_client", "initial_balance": 1000}'

# 4. Submit a job
curl -X POST http://localhost:5000/job/submit \
  -H "Content-Type: application/json" \
  -d '{"client_address": "test_client", "model_type": "linear_regression", "reward": 50}'

# 5. Trigger coordination
curl -X POST http://localhost:5000/network/coordination/trigger

# 6. Check network stats
curl http://localhost:5000/network/stats
```

## 📁 Project Structure

```
decentralized-ai-compute-network/
├── smart_contract.py          # Smart contract implementation
├── worker_node.py              # Worker node implementation
├── client_interface.py         # Client interface
├── network_coordinator.py      # Network coordinator
├── api_server.py               # REST API server
├── dashboard.html              # Web dashboard
├── demo_script.py              # Demo script
└── README.md                   # This file
```


## 🛡️ Security Considerations
- This is a **proof-of-concept** and not production-ready. For production use, consider:
- Proper authentication and authorization
- Secure key management
- Input validation and sanitization
- Rate limiting
- DoS protection
- Secure communication (HTTPS, WSS)
- Smart contract audits
- Byzantine fault tolerance
- Sybil attack prevention

## Known Limitations
- No persistent storage (data lost on restart)
- No actual blockchain integration
- Simplified verification (production needs ZK proofs)
- No worker penalties for failures
- Limited to the local network
- No real cryptocurrency integration

## 🤝 Contributing
- This is an educational proof-of-concept. Feel free to fork and extend!

## 📄License
- MIT License - See LICENSE file for details

## 🙏🏻 Acknowledgments
Inspired by:
- Render Network (decentralized GPU rendering)
- Bittensor (decentralized machine learning)
- Ethereum smart contracts
- IPFS and decentralized storage

---

**Built as a proof-of-concept for decentralized AI compute networks**
