#!/usr/bin/env bash
# GeForce FP32-accumulate cap: measure it on a stock card, no unlock.
# Paste into a RunPod pod terminal (any CUDA "devel" image with nvcc) and run:
#   bash run.sh
# Writes results/<UTC stamp>_<GPU>/ with gemm.csv, telemetry.csv and the GPU inventory.
set -euo pipefail

# Non-interactive SSH does not source the image's profile, so add the CUDA toolkit by hand.
export PATH="/usr/local/cuda/bin:$PATH"
command -v nvcc >/dev/null || { echo "nvcc not found: pick a CUDA devel template"; exit 1; }

work="${WORKDIR:-$HOME/geforce-cap}"
mkdir -p "$work"
cd "$work"

cat > gemm_bench.cu <<'EOF'
// cuBLAS GEMM throughput by input type and accumulate type.
// The ratio fp16_acc32 / fp16_acc16 is the number that matters:
// about 1.0 on datacenter parts, about 0.5 on a capped GeForce part.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t st_ = (x); if (st_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS status %d at line %d\n", (int)st_, __LINE__); exit(1); } } while (0)

enum Scale { F32, F16, I32 };

struct Mode {
  const char* name;
  cudaDataType_t ab, c;
  cublasComputeType_t compute;
  Scale scale;
  size_t ab_size, c_size;
};

static const Mode MODES[] = {
  {"fp16_acc16", CUDA_R_16F,  CUDA_R_16F,  CUBLAS_COMPUTE_16F,           F16, 2, 2},
  {"fp16_acc32", CUDA_R_16F,  CUDA_R_16F,  CUBLAS_COMPUTE_32F,           F32, 2, 2},
  {"bf16_acc32", CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F,           F32, 2, 2},
  {"tf32",       CUDA_R_32F,  CUDA_R_32F,  CUBLAS_COMPUTE_32F_FAST_TF32, F32, 4, 4},
  {"int8_acc32", CUDA_R_8I,   CUDA_R_32I,  CUBLAS_COMPUTE_32I,           I32, 1, 4},  // control
};

struct Shape { const char* name; int m, n, k; };

// Square shapes, then Llama-3.1-8B prefill GEMMs at 4,096 tokens (n = tokens).
static const Shape SHAPES[] = {
  {"sq4096",   4096,  4096,  4096},
  {"sq8192",   8192,  8192,  8192},
  {"sq16384", 16384, 16384, 16384},
  {"qkv",      6144,  4096,  4096},
  {"gate_up", 28672,  4096,  4096},
  {"down",     4096,  4096, 14336},
};

// Random data, not zeros: zeros draw less power and inflate clocks.
static void fill(void* dev, cudaDataType_t t, size_t count, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  size_t bytes = 0;
  std::vector<uint8_t> host;
  if (t == CUDA_R_16F) {
    std::vector<__half> v(count); for (auto& x : v) x = __float2half(u(rng));
    bytes = count * 2; host.assign((uint8_t*)v.data(), (uint8_t*)v.data() + bytes);
  } else if (t == CUDA_R_16BF) {
    std::vector<__nv_bfloat16> v(count); for (auto& x : v) x = __float2bfloat16(u(rng));
    bytes = count * 2; host.assign((uint8_t*)v.data(), (uint8_t*)v.data() + bytes);
  } else if (t == CUDA_R_32F) {
    std::vector<float> v(count); for (auto& x : v) x = u(rng);
    bytes = count * 4; host.assign((uint8_t*)v.data(), (uint8_t*)v.data() + bytes);
  } else {
    std::vector<int8_t> v(count); for (auto& x : v) x = (int8_t)(rng() % 7) - 3;
    bytes = count; host.assign((uint8_t*)v.data(), (uint8_t*)v.data() + bytes);
  }
  CK(cudaMemcpy(dev, host.data(), bytes, cudaMemcpyHostToDevice));
}

int main() {
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  cublasHandle_t h;
  CB(cublasCreate(&h));

  float af = 1.f, bf = 0.f;
  __half ah = __float2half(1.f), bh = __float2half(0.f);
  int32_t ai = 1, bi = 0;

  printf("gpu,mode,shape,m,n,k,tflops_median,tflops_min,tflops_max,iters\n");
  for (const Mode& md : MODES) {
    for (const Shape& s : SHAPES) {
      size_t na = (size_t)s.k * s.m, nb = (size_t)s.k * s.n, nc = (size_t)s.m * s.n;
      void *A, *B, *C;
      CK(cudaMalloc(&A, na * md.ab_size));
      CK(cudaMalloc(&B, nb * md.ab_size));
      CK(cudaMalloc(&C, nc * md.c_size));
      fill(A, md.ab, na, 1);
      fill(B, md.ab, nb, 2);

      const void* alpha = md.scale == F32 ? (const void*)&af : md.scale == F16 ? (const void*)&ah : (const void*)&ai;
      const void* beta  = md.scale == F32 ? (const void*)&bf : md.scale == F16 ? (const void*)&bh : (const void*)&bi;
      // A^T * B layout: the one layout every cuBLAS int8 path accepts.
      auto gemm = [&] {
        CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, s.m, s.n, s.k, alpha,
                        A, md.ab, s.k, B, md.ab, s.k, beta, C, md.c, s.m,
                        md.compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      };

      cudaEvent_t t0, t1;
      CK(cudaEventCreate(&t0));
      CK(cudaEventCreate(&t1));
      for (int i = 0; i < 3; i++) gemm();  // warm up
      CK(cudaDeviceSynchronize());

      // Size each trial to about 2 s so clocks and power reach steady state.
      CK(cudaEventRecord(t0)); gemm(); CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
      float one_ms; CK(cudaEventElapsedTime(&one_ms, t0, t1));
      int iters = std::max(5, (int)(2000.f / std::max(one_ms, 0.01f)));

      double flops = 2.0 * s.m * s.n * s.k * iters;
      std::vector<double> tf;
      for (int trial = 0; trial < 3; trial++) {
        CK(cudaEventRecord(t0));
        for (int i = 0; i < iters; i++) gemm();
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms; CK(cudaEventElapsedTime(&ms, t0, t1));
        tf.push_back(flops / (ms * 1e-3) / 1e12);
      }
      std::sort(tf.begin(), tf.end());
      printf("%s,%s,%s,%d,%d,%d,%.1f,%.1f,%.1f,%d\n", prop.name, md.name, s.name,
             s.m, s.n, s.k, tf[1], tf[0], tf[2], iters);
      fflush(stdout);

      CK(cudaEventDestroy(t0));
      CK(cudaEventDestroy(t1));
      CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C));
      // Idle gap so each configuration is its own segment in telemetry.csv.
      std::this_thread::sleep_for(std::chrono::seconds(3));
    }
  }
  CB(cublasDestroy(h));
  return 0;
}
EOF

nvcc -O3 -std=c++17 -o gemm_bench gemm_bench.cu -lcublas

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
out="results/$(date -u +%Y%m%dT%H%M%SZ)_${gpu}"
mkdir -p "$out"
nvidia-smi -q > "$out/nvidia-smi-q.txt"
nvidia-smi --query-gpu=name,driver_version,power.limit,power.max_limit,clocks.max.sm,clocks.max.mem \
  --format=csv > "$out/gpu.csv"
nvcc --version > "$out/nvcc.txt"

nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
  --format=csv -lms 500 > "$out/telemetry.csv" &
mon=$!
trap 'kill $mon 2>/dev/null || true' EXIT

./gemm_bench | tee "$out/gemm.csv"

echo
echo "== fp16_acc32 / fp16_acc16 (about 0.5 = capped, about 1.0 = full rate)"
awk -F, '$2=="fp16_acc16"{a[$3]=$7} $2=="fp16_acc32"{b[$3]=$7}
  END{for (s in a) if (s in b) printf "%-8s %6.1f / %6.1f TFLOPS = %.2f\n", s, b[s], a[s], b[s]/a[s]}' "$out/gemm.csv"
echo
gzip -c "$out/telemetry.csv" | base64 | wc -c | xargs echo "telemetry bytes (base64 gz):"
echo "Results in $work/$out"
