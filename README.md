================================================================================
 INFERMESH — Production-Grade Multi-Tenant Machine Learning Inference Engine
================================================================================

InferMesh is an open-source, high-performance inference platform built to serve
stateful, multi-tenant ML models at financial-grade reliability. It couples a
Go orchestration layer with a C++ CUDA backend over Unix Domain Sockets, giving
operators per-request compute-second billing, session affinity routing, chaos
engineering hooks, and comprehensive deployment safety logic — all in one cohesive
system.

This repository is open to contributors. See the CONTRIBUTING section below.


________________________________________________________________________________
TABLE OF CONTENTS
________________________________________________________________________________

  1. Why InferMesh
  2. Architecture Overview
  3. Directory Layout
  4. Core Subsystems
  5. Getting Started
  6. Configuration Reference
  7. Running Tests
  8. Contributing
  9. Roadmap
 10. License


________________________________________________________________________________
1. WHY INFERMESH
________________________________________________________________________________

Existing inference servers (Triton, TorchServe, vLLM) solve the single-tenant,
single-model problem. InferMesh targets a harder problem set:

  * Thousands of tenants, each with their own model weights on shared GPU hardware.
  * Stateful models (RNNs, autoregressive decoders) that require requests within
    a session to land on the same worker to preserve KV-cache state.
  * Financial-grade requirements: compute-second billing, deployment freeze windows
    during market open, output perturbation to block model-extraction attacks.
  * Operational completeness: chaos testing hooks, canary p99 regression detection,
    split-brain fencing tokens, TLS hot-rotation — designed in from day one.


________________________________________________________________________________
2. ARCHITECTURE OVERVIEW
________________________________________________________________________________

                   ┌─────────────────────────────────────┐
                   │           Client / SDK               │
                   │   (Retry-After aware, idempotency)   │
                   └───────────────┬─────────────────────┘
                                   │ HTTPS / TLS 1.3
                   ┌───────────────▼─────────────────────┐
                   │          Go Gateway Layer             │
                   │                                       │
                   │  Auth → Registry → Rate Limit →       │
                   │  Quota → Enrichment → Affinity →      │
                   │  Backpressure → Dispatch              │
                   └───────┬───────────────┬──────────────┘
                           │               │
             Unix Domain   │               │  Unix Domain
             Socket (IPC)  │               │  Socket (IPC)
                   ┌───────▼──────┐ ┌──────▼───────┐
                   │  C++ Worker  │ │  C++ Worker  │  ...
                   │  (GPU 0)     │ │  (GPU 1)     │
                   │              │ │              │
                   │  CUDA Streams│ │  TensorPool  │
                   │  Preprocessor│ │  Watchdog    │
                   └──────────────┘ └──────────────┘

Key design choices:

  Go handles orchestration logic: routing, billing, security, lifecycle.
  C++ owns all GPU memory and compute: CUDA streams, tensor pools, preprocessing.
  Communication is via Unix Domain Sockets — no network stack overhead.
  Preprocessing parameters are bundled atomically with model weights —
  the Go layer never touches raw tensor math.


________________________________________________________________________________
3. DIRECTORY LAYOUT
________________________________________________________________________________

  infermesh/
  ├── cpp/
  │   ├── include/            C++ header files
  │   │   ├── Config.h            AppConfig (env-var driven)
  │   │   ├── CudaStreamManager.h CUDA stream pool
  │   │   ├── MemoryManager.h     GPU/CPU memory management
  │   │   ├── Preprocessor.h      Input normalization interface
  │   │   ├── SignalHandler.h     SIGTERM / SIGSEGV handlers
  │   │   ├── TensorPool.h        Thread-safe VRAM buffer pool
  │   │   └── WorkerStatus.h      Status/watchdog file writer
  │   └── src/
  │       ├── CudaStreamManager.cpp
  │       ├── Preprocessor.cpp
  │       ├── SignalHandler.cpp
  │       ├── TensorPool.cpp
  │       ├── WorkerStatus.cpp
  │       └── main_*_test.cpp     Per-subsystem test executables
  │
  ├── go/
  │   ├── cmd/                    Runnable test executables (one per subsystem)
  │   │   ├── affinity_test/
  │   │   ├── backpressure_test/
  │   │   ├── canary_test/
  │   │   ├── chaos_test/
  │   │   ├── compute_billing_test/
  │   │   ├── enrichment_test/
  │   │   ├── fencing_test/
  │   │   ├── freeze_test/
  │   │   ├── hot_reload_test/
  │   │   ├── idempotency_test/
  │   │   ├── observability_test/
  │   │   ├── orchestrator_test/
  │   │   ├── quota_test/
  │   │   ├── replication_test/
  │   │   ├── security_test/
  │   │   ├── suspend_test/
  │   │   ├── tls_rotator_test/
  │   │   └── watchdog_test/
  │   │
  │   ├── pkg/
  │   │   ├── chaos/              Chaos engineering injection API
  │   │   ├── config/             Hot-reload config manager
  │   │   ├── deployment/         Orchestrator, fencing, freeze, canary, replication
  │   │   ├── gateway/            Auth, billing, enrichment, idempotency, observability,
  │   │   │                       quota, security, TLS rotator, rate limiter, pipeline
  │   │   ├── registry/           Tenant registry + lifecycle (bootstrap, suspend, resume)
  │   │   ├── router/             Affinity routing, backpressure tracker
  │   │   └── worker/             Process launcher, watchdog monitor
  │   │
  │   ├── sdk/
  │   │   └── client.go           Client SDK (Retry-After aware, idempotency keys)
  │   │
  │   └── go.mod
  │
  └── proto/                  Protobuf definitions (Go ↔ C++ contract)


________________________________________________________________________________
4. CORE SUBSYSTEMS
________________________________________________________________________________

--- C++ BACKEND ---

  TensorPool
    Pre-allocates a fixed number of GPU memory buffers at startup. Workers
    acquire buffers from the pool before inference and release them after.
    Prevents per-request cudaMalloc/Free overhead and VRAM fragmentation.
    Blocks (with configurable timeout) rather than over-allocating on spikes.

  CudaStreamManager / PipelinedEngine
    Maintains a pool of CUDA streams. Overlaps Host-to-Device memory copies
    with GPU compute kernels using cudaMemcpyAsync, eliminating idle GPU time
    between sequential requests.

  Preprocessor
    Normalization parameters (means, std_devs) are loaded from a sidecar config
    file bundled atomically with model weights. The Go layer never touches these
    values, preventing misconfiguration from CI/CD pipelines.

  SignalHandler + WorkerStatus
    SIGTERM/SIGINT trigger a graceful shutdown: drain in-flight requests, release
    VRAM, close sockets. SIGSEGV/SIGABRT/SIGILL write "CRASHED" to the status
    file and re-raise the signal to generate an OS core dump.
    UpdateWatchdog() writes a heartbeat epoch timestamp every N inferences so
    the Go supervisor can detect frozen (not just dead) processes.

--- GO ORCHESTRATION LAYER ---

  Request Pipeline (pkg/gateway/pipeline.go)
    Eight-stage sequential pipeline: TLS termination → IP rate limit →
    request parsing → JWT auth → registry lookup → tenant rate limit →
    request ID assignment → dispatch.

  Session Affinity Router (pkg/router/affinity.go)
    Maps session_id to a specific worker to preserve RNN / autoregressive
    KV-cache state. On worker death, applies per-tenant policy:
      PolicyRebuildState  — transparently migrate session to a new worker.
      PolicyErrorSessionLost — return ErrSessionLost to let the client reset.

  Backpressure Propagation (pkg/router/backpressure.go)
    Computes Retry-After header value as:
      (queue_depth × ewma_latency_ms) / active_workers / 1000
    Client SDK honours this value, replacing its own local backoff strategy.

  Idempotency (pkg/gateway/idempotency.go)
    Idempotency-Key header deduplicates retries at the gateway. Cached
    responses are replayed without hitting C++ workers. Error responses
    (5xx) are not cached. TTL is configurable per deployment.

  Quota Enforcement (pkg/gateway/quota.go)
    Monthly request-count quota distinct from the per-second rate limit.
    Returns HTTP 402 (not 429) when the monthly cap is hit. Billing cycle
    anchors to the tenant's signup day, not the calendar month.

  Compute-Second Billing (pkg/gateway/compute_billing.go)
    Accumulates C++-reported GPU microseconds per tenant per billing period.
    Uses the same period-key structure as quota counters so both land in the
    same durable store. 100 concurrent writes produce exact totals — no drift.

  Feature Store Enrichment (pkg/gateway/enrichment.go)
    Between request parsing and C++ dispatch, Go queries an external feature
    store for computed features (market data, account history). Three timeout
    policies per tenant:
      TimeoutReject        — abort inference if the store is slow.
      TimeoutProceedPartial — continue with zero-filled missing features.
      TimeoutQueue         — hold the request and retry until the store recovers.

  Tenant Suspension (pkg/registry/bootstrap.go)
    SuspendTenant() sets status = SUSPENDED. EvaluateRequestReadiness() blocks
    all routing immediately. Workers stay alive (VRAM preserved). ResumeTenant()
    restores traffic in zero milliseconds — no worker restart penalty.

  Deployment Orchestrator (pkg/deployment/orchestrator.go)
    Deployment manifests specify required_cpp_binary_version. Go checks the
    running engine version before deploying model weights. If the engine is
    too old: deploy binary → wait for health checks → deploy weights. Deploying
    in the wrong order causes runtime type mismatches and is structurally
    prevented by the DAG evaluation.

  Deployment Freeze Windows (pkg/deployment/freeze.go)
    Configurable per-tenant blackout periods (e.g., NYSE open 08:00–10:00 ET
    every Monday). All deployment attempts — including CI/CD webhooks — are
    rejected during a freeze. Emergency bypass requires a separate admin token
    that is logged to the audit trail.

  Split-Brain Fencing Tokens (pkg/deployment/fencing.go)
    Distributed lock returns a monotonically increasing fencing token (etcd /
    Redlock style). The target resource rejects writes bearing a token older
    than the highest token it has seen, preventing zombie primary nodes from
    overwriting work done by the new elected leader.

  Canary P99 Regression Detection (pkg/deployment/canary.go)
    During canary rollout, Go tracks p99 latency for baseline (v1) and canary
    (v2) workers from actual traffic samples. If p99 regression > configured
    threshold (default 20%), the rollout halts. An explicit operator override
    token is required to acknowledge and continue.

  Async Cross-Region Replication (pkg/deployment/replication.go)
    On every primary-region model deployment, Go fires background goroutines
    to copy artifact files to all secondary regions. Secondary regions maintain
    warm standby copies. Failover activates in seconds, not hours.

  TLS Hot Rotation (pkg/gateway/tls_rotator.go)
    Polls the certificate file path on a configurable interval. When Certbot or
    cert-manager writes a new PEM file, Go atomically swaps the tls.Certificate
    pointer. New TLS handshakes use the new certificate immediately. In-flight
    connections complete on the old certificate undisturbed.

  Hot-Reload Configuration (pkg/config/hot_reload.go)
    Validates incoming config before applying it. Invalid configs (MaxWorkers=0,
    MinWorkers > MaxWorkers, missing TLS path) are rejected and the previous
    healthy config is retained. Valid configs perform surgical subsystem updates:
    only the affected rate limiter, auto-scaler, or TLS layer is notified.

  Observability Cardinality Control (pkg/gateway/observability.go)
    Real-time Prometheus metrics use bounded labels only (worker_id,
    model_version, error_type). Per-tenant counters are accumulated in RAM and
    flushed periodically to a high-cardinality TSDB (InfluxDB, ClickHouse).
    Prometheus is never exposed to tenant_id labels — no TSDB poisoning.

  Security Interceptor (pkg/gateway/security.go)
    Output perturbation: adds configurable Gaussian noise to confidence scores
    before returning to clients. Noise is small enough for human decision-making
    but corrupts automated gradient extraction. Anomaly detection: tracks feature
    vector deltas per tenant; systematic linear sweeps (grid search attacks)
    trigger an alert and can auto-trigger SuspendTenant().

  Chaos Engineering API (pkg/chaos/injector.go)
    Admin-only API (protected by mTLS) with four injection endpoints:
      KillWorker(id)              — SIGKILL a specific GPU process.
      InjectSocketLatency(ms)     — delay Unix socket I/O by N milliseconds.
      SimulateQueueSpike(depth)   — bloat the dispatch queue counter.
      SimulateS3Failure(bool)     — sever the cross-region storage link.
    All endpoints are no-ops unless EnableChaosAPI = true in config.
    This flag must be false in production environments.


________________________________________________________________________________
5. GETTING STARTED
________________________________________________________________________________

Prerequisites
  Go 1.22 or higher          https://go.dev/dl/
  GCC / Clang with C++17     apt install build-essential  (Linux)
  NVIDIA CUDA Toolkit 12+    https://developer.nvidia.com/cuda-downloads
  (Optional) etcd             For production fencing token consensus

Clone the repository
  git clone https://github.com/your-org/infermesh.git
  cd infermesh

Go module setup
  cd go
  go mod tidy

Build the C++ workers
  cd cpp
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j$(nproc)

Run a single test executable (no external dependencies)
  cd go
  go run cmd/compute_billing_test/main.go
  go run cmd/affinity_test/main.go
  go run cmd/chaos_test/main.go

Environment variables consumed by C++ workers
  INFERMESH_SOCKET_PATH    Unix socket path for this worker instance
  INFERMESH_MODEL_PATH     Path to the model weights directory
  INFERMESH_GPU_ID         CUDA device index (0-based)
  INFERMESH_WORKER_ID      Unique integer identifier for this process


________________________________________________________________________________
6. CONFIGURATION REFERENCE
________________________________________________________________________________

Go orchestrator configuration (AppConfig struct — pkg/config/hot_reload.go):

  Field              Type     Default   Description
  ─────────────────────────────────────────────────────────────────────────────
  MaxWorkers         int      10        Hard upper bound on C++ worker processes
  MinWorkers         int      1         Guaranteed always-warm worker count
  RateLimitRPS       int      1000      Per-tenant requests per second limit
  TLSCertPath        string   required  Path to PEM certificate file

All fields are hot-reloadable at runtime. Changes pass through the validation
layer. Invalid values are rejected silently and the previous config is retained.

Tenant-level overrides (set via Registry API):

  Field                  Description
  ─────────────────────────────────────────────────────────────────────────────
  MonthlyQuota           Max requests per billing cycle (HTTP 402 on exhaustion)
  EnrichmentTimeout      Max wait for feature store before applying timeout policy
  EnrichmentPolicy       TimeoutReject | TimeoutProceedPartial | TimeoutQueue
  FreezeWindows          Array of {Day, StartTime, EndTime, Timezone} structs
  EmergencyToken         Bypass token for freeze window overrides (audit-logged)
  SessionAffinityPolicy  PolicyRebuildState | PolicyErrorSessionLost
  PerturbationNoise      Noise scale applied to output confidence scores (0–1)
  AnomalyWindowSize      Number of requests tracked for sweep detection


________________________________________________________________________________
7. RUNNING TESTS
________________________________________________________________________________

Each cmd/ subdirectory is a self-contained test executable with its own main()
function. No test framework is required — tests print [PASS] / [FAIL] and exit
with code 0 or 1.

Run all Go tests:
  cd go
  for dir in cmd/*/; do
    echo "--- $dir ---"
    go run "$dir"main.go
  done

Run a specific subsystem test:
  go run cmd/canary_test/main.go
  go run cmd/fencing_test/main.go
  go run cmd/freeze_test/main.go
  go run cmd/tls_rotator_test/main.go

Run C++ tests (after building):
  ./build/main_tensorpool_test
  ./build/main_cuda_stream_test
  ./build/main_preprocessor_test

Chaos tests must be run explicitly as they require EnableChaosAPI = true:
  go run cmd/chaos_test/main.go


________________________________________________________________________________
8. CONTRIBUTING
________________________________________________________________________________

InferMesh welcomes contributors. Here is how to get involved:

Fork and Branch
  Fork the repository on GitHub, then create a feature branch:
    git checkout -b feature/your-feature-name

Code Style
  Go code follows standard gofmt formatting. Run before committing:
    gofmt -w ./...
    go vet ./...

  C++ code follows the Google C++ Style Guide with two exceptions:
    * Indentation: 4 spaces (not 2)
    * Brace style: same line for functions

Commit Messages
  Use the following prefix convention:
    feat:    New capability
    fix:     Bug fix
    perf:    Performance improvement
    refactor: Structural change with no behaviour change
    test:    New or updated tests
    docs:    Documentation only

  Full example:
    feat(gateway): add per-tenant output perturbation noise config

Testing Requirements
  All new Go subsystems require a corresponding cmd/*_test/main.go executable
  demonstrating at minimum:
    * The happy path (normal operation)
    * At least one explicit failure/rejection case
    * Concurrency safety if the subsystem uses shared state

  All new C++ subsystems require a corresponding main_*_test.cpp.

Pull Request Process
  1. Ensure all tests in cmd/ pass with exit code 0.
  2. Ensure go vet ./... produces no output.
  3. Update the README (this file) if you introduce a new subsystem.
  4. Open a PR against the main branch with a clear description of:
       * What problem the change solves
       * Any architectural decisions made and alternatives considered
       * Whether production deployment requires a configuration change

Areas Actively Seeking Contributors
  * Kubernetes operator for automated worker pool lifecycle management
  * Prometheus exporter wiring the observability layer to real metrics endpoints
  * etcd integration replacing MockConsensusStore in the fencing token system
  * Redis integration replacing in-memory mocks in quota and billing stores
  * gRPC transport layer between Go and C++ (replacing Unix socket JSON encoding)
  * Integration test harness running the full Go + C++ stack in a container
  * Benchmarks for the TensorPool under high concurrency
  * Documentation for the proto/ contract between Go and C++

Issue Labels
  good-first-issue    Beginner-friendly, well-scoped tasks
  help-wanted         Larger tasks where external help is appreciated
  performance         Optimization work on hot paths
  security            Any change touching auth, TLS, or output perturbation
  breaking-change     Changes to the Go ↔ C++ protocol or public API


________________________________________________________________________________
9. ROADMAP
________________________________________________________________________________

v0.2 — Integration Layer
  [ ] Replace all MockConsensusStore instances with etcd client
  [ ] Replace MockQuotaStore / MockComputeLedger with Redis INCRBY pipeline
  [ ] gRPC transport between Go gateway and C++ workers
  [ ] Kubernetes CRD: InferMeshTenant (auto-provisions Registry entries)

v0.3 — Observability
  [ ] Prometheus /metrics endpoint wired to LowCardinalityMetrics
  [ ] Grafana dashboard provisioning (JSON model included in repo)
  [ ] Distributed tracing (OpenTelemetry) across Go pipeline stages
  [ ] Structured JSON logging replacing fmt.Printf throughout

v0.4 — Multi-Region
  [ ] Terraform module for multi-region AWS deployment (us-east-1, eu-west-1)
  [ ] Automated cert-manager integration for TLS hot rotation
  [ ] Health check endpoint consumed by AWS Route53 / Cloudflare failover rules

v1.0 — Production Certification
  [ ] Full integration test suite (Go + C++ in Docker Compose)
  [ ] Load test: 10,000 concurrent tenants, p99 < 50ms gateway overhead
  [ ] Security audit of the mTLS admin API surface
  [ ] Formal chaos experiment playbook (validated against all FailureModes)


________________________________________________________________________________
10. LICENSE
________________________________________________________________________________

InferMesh is released under the Apache License 2.0.
See the LICENSE file in the root of this repository for the full text.

Copyright 2026 InferMesh Contributors.
