#include <torch/extension.h>

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace vllm_hcu_step5_packed_masked {

using bf16 = __hip_bfloat16;
using int8x8 = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;
using int32x4 = __attribute__((__vector_size__(4 * sizeof(int32_t)))) int32_t;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using uintx4 = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using bf16x8_bits = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;

constexpr int kMmacM = 16;
constexpr int kBlockM = 64;
constexpr int kBlockN = 256;
constexpr int kBlockK = 128;
constexpr int kKPerBPayloadTile = 64;
constexpr int kNumWarps = 4;
constexpr int kWarpSize = 64;
constexpr int kThreads = kNumWarps * kWarpSize;
constexpr int kMmacN = 16;
constexpr int kRowsPerMSubtile = 16;
constexpr int kElementsPerLane = 16;
constexpr int kMSubtiles = kBlockM / kRowsPerMSubtile;
constexpr int kN32Cols = 32;
constexpr int kNPairsPerWave = kBlockN / (kNumWarps * kN32Cols);
constexpr int kMSubtileBytes = kRowsPerMSubtile * kBlockK;
constexpr int kK64TileBytes = kRowsPerMSubtile * kKPerBPayloadTile;
constexpr int kColGroupBytes = kRowsPerMSubtile * kElementsPerLane;
constexpr int kLaneBytes = kElementsPerLane;
constexpr int kAStageBytes = kBlockM * kBlockK;
constexpr int kAStageCount = 2;
constexpr int kASharedBytes = kAStageBytes * kAStageCount;
constexpr int kScaleSharedFloats = kBlockM + kBlockN;
constexpr int kScaleSharedBytes = kScaleSharedFloats * sizeof(float);
constexpr int kSharedBytes = kASharedBytes + kScaleSharedBytes;
constexpr int kALoaderWaves = kNumWarps;
static_assert(kMSubtiles == 4, "This kernel expects BM64 as four M16 subtiles");
static_assert(kNPairsPerWave == 2, "This kernel expects two N32 pairs per wave for BN256");
static_assert(kAStageBytes == 8192, "A LDS stage should be 8KB");
static_assert(kASharedBytes == 16384, "double-buffered A LDS should be 16KB");
static_assert(kScaleSharedBytes == 1280, "scale LDS should be 1.25KB for BM64/BN256");
static_assert(kSharedBytes == 17664, "total LDS should be 17.25KB");

union Fp8x16Fragment {
  int8x8 i8[2];
  int32x4 i32;
};

using AccFragment = floatx4;

struct AccPairFragment {
  AccFragment even;
  AccFragment odd;
};

struct BPairFragment {
  Fp8x16Fragment even;
  Fp8x16Fragment odd;
};

__device__ __forceinline__ AccFragment zero_acc_fragment() {
  return AccFragment{0.0f, 0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ AccPairFragment zero_acc_pair_fragment() {
  return AccPairFragment{zero_acc_fragment(), zero_acc_fragment()};
}

__device__ __forceinline__ int ceil_div_int(int x, int y) {
  return (x + y - 1) / y;
}

__device__ __forceinline__ floatx4 mmac_fp8_16x16x32(const int8x8& a,
                                                       const int8x8& b,
                                                       floatx4 acc) {
#if defined(__gfx938__)
  return __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(a, b, acc, 0, 0);
#elif defined(__gfx92a__)
  return __builtin_hcu_mmac_f32_16x16x32_fp8_fp8(a, b, acc);
#else
  return acc;
#endif
}

__device__ __forceinline__ uint16_t bf16_bits(float value) {
  union FloatBits {
    float f;
    uint32_t u;
  };
  FloatBits bits;
  bits.f = value;
  const uint32_t lsb = (bits.u >> 16) & 1u;
  bits.u += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits.u >> 16);
}

__device__ __forceinline__ void wait_vmcnt_barrier() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(0)\n\ts_barrier\n\t" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt_lgkmcnt_barrier() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier\n\t" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void barrier() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_barrier\n\t" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt_zero() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(0)\n\t" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt_2() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(2)\n\t" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt_4() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(4)\n\t" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void wait_lgkmcnt_zero() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt lgkmcnt(0)\n\t" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void wait_vmcnt_one() {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(1)\n\t" ::: "memory");
  __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ void buffer_load_dwordx4_to_lds(
    char* __restrict__ lds_base,
    int lds_byte_offset,
    const uintx4& global_ptr,
    int global_byte_offset) {
  char* lds_ptr = lds_base + lds_byte_offset;
  __builtin_amdgcn_sched_barrier(0);
  __builtin_amdgcn_raw_buffer_load_lds(
      global_ptr,
      (__attribute__((address_space(3))) int*)lds_ptr,
      16,
      global_byte_offset,
      0,
      0,
      0);
  __builtin_amdgcn_sched_barrier(0);
}

__device__ __forceinline__ uintx4 make_raw_buffer_resource(
    const void* __restrict__ ptr) {
  uintx4 global_ptr;
  *reinterpret_cast<uint64_t*>(&global_ptr) = reinterpret_cast<uint64_t>(ptr);
  global_ptr[1] += 0x40800000;
  global_ptr[2] = 0x80000000;
  global_ptr[3] = 0x00020000;
  return global_ptr;
}

__device__ __forceinline__ void stage_output_scales_to_lds(
    float* __restrict__ scale_tile_lds,
    const float* __restrict__ a_scale,
    const float* __restrict__ b_scale,
    const uintx4& a_scale_global_ptr,
    const uintx4& b_scale_global_ptr,
    int expert,
    int m_block_start,
    int n_block_base,
    int valid_m,
    int size_m,
    int size_n,
    int thread_id) {
#pragma unroll
  for (int local_row = thread_id; local_row < kBlockM; local_row += kThreads) {
    const int row = m_block_start + local_row;
    scale_tile_lds[local_row] =
        (row < valid_m && row < size_m)
            ? a_scale[static_cast<int64_t>(expert) * size_m + row]
            : 0.0f;
  }

#pragma unroll
  for (int local_col = thread_id; local_col < kBlockN; local_col += kThreads) {
    const int col = n_block_base + local_col;
    scale_tile_lds[kBlockM + local_col] =
        b_scale[static_cast<int64_t>(expert) * size_n + col];
  }
}

__device__ __forceinline__ Fp8x16Fragment ds_read_b128_fp8(
    const int8_t* __restrict__ lds_base,
    int byte_offset) {
  Fp8x16Fragment data;
  data.i32 = *reinterpret_cast<const int32x4*>(lds_base + byte_offset);
  return data;
}

__device__ __forceinline__ Fp8x16Fragment buffer_load_dwordx4_to_vgpr_fp8(
    const uintx4& global_ptr,
    int byte_offset) {
  Fp8x16Fragment data;
  uintx4 data_words;
  asm volatile("buffer_load_dwordx4 %0,%1,%2,0 offen offset:0\n"
               : "=v"(data_words)
               : "v"(byte_offset), "s"(global_ptr)
               : "memory");
  data.i32 = *reinterpret_cast<int32x4*>(&data_words);
  return data;
}

__device__ __forceinline__ Fp8x16Fragment load_b_fragment(
    const uintx4& global_ptr,
    int k_pack_iter,
    int n32_tile_count,
    int n_tile_id,
    int n_pair,
    int pair_id,
    int wave_id,
    int b_k_group,
    int row_id) {
  const int wave_n32_index = n_pair * kNumWarps + wave_id;
  const int n32_band = n_tile_id * (kBlockN / kN32Cols) + wave_n32_index;
  const int64_t b_offset =
      (((((static_cast<int64_t>(k_pack_iter) * n32_tile_count +
           n32_band) *
              2 +
          pair_id) *
             4 +
         b_k_group) *
            16 +
        row_id) *
       16);
  return buffer_load_dwordx4_to_vgpr_fp8(
      global_ptr, static_cast<int>(b_offset));
}

__device__ __forceinline__ int compute_b_fragment_byte_offset(
    int k_pack_iter,
    int n32_tile_count,
    int n_tile_id,
    int n_pair,
    int pair_id,
    int wave_id,
    int b_k_group,
    int row_id) {
  const int wave_n32_index = n_pair * kNumWarps + wave_id;
  const int n32_band = n_tile_id * (kBlockN / kN32Cols) + wave_n32_index;
  const int64_t b_offset =
      (((((static_cast<int64_t>(k_pack_iter) * n32_tile_count +
           n32_band) *
              2 +
          pair_id) *
             4 +
         b_k_group) *
            16 +
        row_id) *
       16);
  return static_cast<int>(b_offset);
}

__device__ __forceinline__ BPairFragment load_b_pair_fragment(
    const uintx4& global_ptr,
    int k_pack_iter,
    int n32_tile_count,
    int n_tile_id,
    int n_pair,
    int wave_id,
    int b_k_group,
    int row_id) {
  BPairFragment b_pair;
  b_pair.even = load_b_fragment(
      global_ptr, k_pack_iter, n32_tile_count, n_tile_id, n_pair, 0,
      wave_id, b_k_group, row_id);
  b_pair.odd = load_b_fragment(
      global_ptr, k_pack_iter, n32_tile_count, n_tile_id, n_pair, 1,
      wave_id, b_k_group, row_id);
  return b_pair;
}

template <int kNPairs>
__device__ __forceinline__ void load_b_pairs(
    BPairFragment (&b_pair)[kNPairs],
    const uintx4& global_ptr,
    int k_pack_iter,
    int n32_tile_count,
    int n_tile_id,
    int wave_id,
    int b_k_group,
    int row_id) {
#pragma unroll
  for (int n_pair = 0; n_pair < kNPairs; ++n_pair) {
    b_pair[n_pair] = load_b_pair_fragment(
        global_ptr,
        k_pack_iter,
        n32_tile_count,
        n_tile_id,
        n_pair,
        wave_id,
        b_k_group,
        row_id);
  }
}

template <int kNPairs>
__device__ __forceinline__ void load_b_pairs_2k64_precompute_offsets(
    BPairFragment (&b_first)[kNPairs],
    BPairFragment (&b_second)[kNPairs],
    const uintx4& global_ptr,
    int k_pack_iter_first,
    int n32_tile_count,
    int n_tile_id,
    int wave_id,
    int b_k_group,
    int row_id) {
  static_assert(kNPairs == kNPairsPerWave,
                "This kernel precomputes the full per-wave BN256 B payload");

  int first_even[kNPairs];
  int first_odd[kNPairs];
  int second_even[kNPairs];
  int second_odd[kNPairs];

#pragma unroll
  for (int n_pair = 0; n_pair < kNPairs; ++n_pair) {
    first_even[n_pair] = compute_b_fragment_byte_offset(
        k_pack_iter_first, n32_tile_count, n_tile_id,
        n_pair, 0, wave_id, b_k_group, row_id);
    first_odd[n_pair] = compute_b_fragment_byte_offset(
        k_pack_iter_first, n32_tile_count, n_tile_id,
        n_pair, 1, wave_id, b_k_group, row_id);
    second_even[n_pair] = compute_b_fragment_byte_offset(
        k_pack_iter_first + 1, n32_tile_count, n_tile_id,
        n_pair, 0, wave_id, b_k_group, row_id);
    second_odd[n_pair] = compute_b_fragment_byte_offset(
        k_pack_iter_first + 1, n32_tile_count, n_tile_id,
        n_pair, 1, wave_id, b_k_group, row_id);
  }

  __builtin_amdgcn_sched_barrier(0);

#pragma unroll
  for (int n_pair = 0; n_pair < kNPairs; ++n_pair) {
    b_first[n_pair].even = buffer_load_dwordx4_to_vgpr_fp8(
        global_ptr, first_even[n_pair]);
    b_first[n_pair].odd = buffer_load_dwordx4_to_vgpr_fp8(
        global_ptr, first_odd[n_pair]);
  }

#pragma unroll
  for (int n_pair = 0; n_pair < kNPairs; ++n_pair) {
    b_second[n_pair].even = buffer_load_dwordx4_to_vgpr_fp8(
        global_ptr, second_even[n_pair]);
    b_second[n_pair].odd = buffer_load_dwordx4_to_vgpr_fp8(
        global_ptr, second_odd[n_pair]);
  }

  __builtin_amdgcn_sched_barrier(0);
}

template <int kBlockM_, int kBlockK_>
__device__ __forceinline__ void load_a_tile_to_lds_4w(
    int8_t* __restrict__ a_lds,
    const int8_t* __restrict__ input,
    int64_t a_expert_offset,
    int m_block_start,
    int k_block_start,
    int valid_m,
    int size_m,
    int size_k,
    int wave_id,
    int lane_id,
    bool tile_has_valid_rows) {
  static_assert(kBlockM_ == kBlockM, "This kernel expects BM64");
  static_assert(kBlockK_ == kBlockK, "This kernel expects BK128");

  if (wave_id >= kALoaderWaves) {
    return;
  }

  const int row_id = lane_id & 15;
  const int col_group = lane_id >> 4;
  const bool full_m_tile =
      tile_has_valid_rows && (m_block_start + kBlockM <= size_m) &&
      (k_block_start + kBlockK <= size_k);

  if (full_m_tile) {
    const int8_t* tile_input = input + a_expert_offset + static_cast<int64_t>(m_block_start) * size_k;
    uintx4 global_ptr;
    *reinterpret_cast<uint64_t*>(&global_ptr) = reinterpret_cast<uint64_t>(tile_input);
    global_ptr[1] += 0x41000000;
    global_ptr[2] = static_cast<uint32_t>(kBlockM * size_k + 16);
    global_ptr[3] = 0x00020000;

#pragma unroll
    for (int m_round = 0; m_round < kMSubtiles / kALoaderWaves; ++m_round) {
      const int m_loop = m_round * kALoaderWaves + wave_id;
      const int local_row = m_loop * kRowsPerMSubtile + row_id;
#pragma unroll
      for (int k64 = 0; k64 < kBlockK; k64 += kKPerBPayloadTile) {
        const int local_k = k64 + col_group * kElementsPerLane;
        const int lds_byte_offset =
            m_loop * kMSubtileBytes + (k64 / kKPerBPayloadTile) * kK64TileBytes;
        const int global_byte_offset =
            static_cast<int>(static_cast<int64_t>(local_row) * size_k +
                             k_block_start + local_k);
        buffer_load_dwordx4_to_lds(
            reinterpret_cast<char*>(a_lds), lds_byte_offset, global_ptr,
            global_byte_offset);
      }
    }
    return;
  }

#pragma unroll
  for (int m_round = 0; m_round < kMSubtiles / kALoaderWaves; ++m_round) {
    const int m_loop = m_round * kALoaderWaves + wave_id;
    const int local_row = m_loop * kRowsPerMSubtile + row_id;
    const int global_row = m_block_start + local_row;
#pragma unroll
    for (int k64 = 0; k64 < kBlockK; k64 += kKPerBPayloadTile) {
      const int local_k = k64 + col_group * kElementsPerLane;
      const int global_k = k_block_start + local_k;
      const int lds_byte_offset =
          m_loop * kMSubtileBytes + (k64 / kKPerBPayloadTile) * kK64TileBytes +
          col_group * kColGroupBytes + row_id * kLaneBytes;
      Fp8x16Fragment data;
      data.i32 = int32x4{0, 0, 0, 0};
      if (tile_has_valid_rows && global_row < valid_m && global_row < size_m &&
          global_k + 15 < size_k) {
        const int64_t a_offset =
            a_expert_offset + static_cast<int64_t>(global_row) * size_k + global_k;
        data.i32 = *reinterpret_cast<const int32x4*>(input + a_offset);
      }
      *reinterpret_cast<int32x4*>(a_lds + lds_byte_offset) = data.i32;
    }
  }
}

template <int kMSubtiles_, int kNPairs>
__device__ __forceinline__ void accumulate_one_k64_tile(
    const int8_t* __restrict__ a_lds_current,
    int k64,
    int m_block_start,
    int valid_m,
    int row_id,
    int b_k_group,
    const BPairFragment (&b_pair)[kNPairs],
    AccPairFragment (&acc)[kMSubtiles_][kNPairs]) {
#pragma unroll
  for (int m_loop = 0; m_loop < kMSubtiles_; ++m_loop) {
    const int a_lds_byte_offset =
        m_loop * kMSubtileBytes + (k64 / kKPerBPayloadTile) * kK64TileBytes +
        b_k_group * kColGroupBytes + row_id * kLaneBytes;
    Fp8x16Fragment a_frag =
        ds_read_b128_fp8(a_lds_current, a_lds_byte_offset);
    if (m_block_start + m_loop * kMmacM < valid_m) {
#pragma unroll
      for (int n_pair = 0; n_pair < kNPairs; ++n_pair) {
        acc[m_loop][n_pair].even = mmac_fp8_16x16x32(
            a_frag.i8[0], b_pair[n_pair].even.i8[0],
            acc[m_loop][n_pair].even);
        acc[m_loop][n_pair].even = mmac_fp8_16x16x32(
            a_frag.i8[1], b_pair[n_pair].even.i8[1],
            acc[m_loop][n_pair].even);
        acc[m_loop][n_pair].odd = mmac_fp8_16x16x32(
            a_frag.i8[0], b_pair[n_pair].odd.i8[0],
            acc[m_loop][n_pair].odd);
        acc[m_loop][n_pair].odd = mmac_fp8_16x16x32(
            a_frag.i8[1], b_pair[n_pair].odd.i8[1],
            acc[m_loop][n_pair].odd);
      }
    }
  }
}

template <int kMSubtiles_>
__device__ __forceinline__ void load_a_fragments_for_k64_tile(
    Fp8x16Fragment (&a_frag)[kMSubtiles_],
    const int8_t* __restrict__ a_lds_current,
    int k64,
    int row_id,
    int b_k_group) {
#pragma unroll
  for (int m_loop = 0; m_loop < kMSubtiles_; ++m_loop) {
    const int a_lds_byte_offset =
        m_loop * kMSubtileBytes + (k64 / kKPerBPayloadTile) * kK64TileBytes +
        b_k_group * kColGroupBytes + row_id * kLaneBytes;
    a_frag[m_loop] = ds_read_b128_fp8(a_lds_current, a_lds_byte_offset);
  }
}

template <int kMSubtiles_, int kNPairs>
__device__ __forceinline__ void accumulate_one_k64_tile_from_a_fragments(
    const Fp8x16Fragment (&a_frag)[kMSubtiles_],
    int m_block_start,
    int valid_m,
    const BPairFragment (&b_pair)[kNPairs],
    AccPairFragment (&acc)[kMSubtiles_][kNPairs]) {
#pragma unroll
  for (int m_loop = 0; m_loop < kMSubtiles_; ++m_loop) {
    if (m_block_start + m_loop * kMmacM < valid_m) {
#pragma unroll
      for (int n_pair = 0; n_pair < kNPairs; ++n_pair) {
        acc[m_loop][n_pair].even = mmac_fp8_16x16x32(
            a_frag[m_loop].i8[0], b_pair[n_pair].even.i8[0],
            acc[m_loop][n_pair].even);
        acc[m_loop][n_pair].even = mmac_fp8_16x16x32(
            a_frag[m_loop].i8[1], b_pair[n_pair].even.i8[1],
            acc[m_loop][n_pair].even);
        acc[m_loop][n_pair].odd = mmac_fp8_16x16x32(
            a_frag[m_loop].i8[0], b_pair[n_pair].odd.i8[0],
            acc[m_loop][n_pair].odd);
        acc[m_loop][n_pair].odd = mmac_fp8_16x16x32(
            a_frag[m_loop].i8[1], b_pair[n_pair].odd.i8[1],
            acc[m_loop][n_pair].odd);
      }
    }
  }
}

template <int kMLoop>
__device__ __forceinline__ void store_full_m_subtile_pair(
    const AccPairFragment (&acc)[kMSubtiles][kNPairsPerWave],
    int n_pair,
    const float* __restrict__ b_scale_frag,
    const float (&a_scale_frag)[kMSubtiles],
    const int64_t (&c_offset_frag)[kMSubtiles],
    bf16* __restrict__ c,
    int store_col_base) {
  const float a_s = a_scale_frag[kMLoop];
  bf16x8_bits out_bits;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float even_value =
        static_cast<float>(acc[kMLoop][n_pair].even[r]) * a_s * b_scale_frag[r * 2];
    const float odd_value =
        static_cast<float>(acc[kMLoop][n_pair].odd[r]) * a_s * b_scale_frag[r * 2 + 1];
    out_bits[r * 2] = bf16_bits(even_value);
    out_bits[r * 2 + 1] = bf16_bits(odd_value);
  }
  *reinterpret_cast<bf16x8_bits*>(c + c_offset_frag[kMLoop] + store_col_base) =
      out_bits;
}

template <int kMLoop>
__device__ __forceinline__ void store_tail_m_subtile_pair(
    const AccPairFragment (&acc)[kMSubtiles][kNPairsPerWave],
    int n_pair,
    const float* __restrict__ b_scale_frag,
    const float (&a_scale_frag)[kMSubtiles],
    const int64_t (&c_offset_frag)[kMSubtiles],
    bf16* __restrict__ c,
    int store_col_base,
    int row_id,
    int tail_rows) {
  if (row_id < tail_rows) {
    store_full_m_subtile_pair<kMLoop>(
        acc, n_pair, b_scale_frag, a_scale_frag, c_offset_frag, c, store_col_base);
  }
}

__global__ __launch_bounds__(kThreads, 1) void grouped_w8a8_fp8_nt_masked_bm64_bn256_bk128_pair32_persistent_a_double_buffer_writeback16_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ packed_b,
    const float* __restrict__ a_scale,
    const float* __restrict__ b_scale,
    const int32_t* __restrict__ masked_m,
    bf16* __restrict__ c,
    int experts,
    int size_m,
    int size_n,
    int size_k,
    int n_tiles,
    bool enable_overlap,
    int32_t* signal) {
  extern __shared__ int8_t a_lds[];
  float* __restrict__ scale_tile_lds =
      reinterpret_cast<float*>(a_lds + kASharedBytes);
  float* __restrict__ a_scale_tile_lds = scale_tile_lds;
  float* __restrict__ b_scale_tile_lds = scale_tile_lds + kBlockM;

  const int lane_id = threadIdx.x & 63;
  const int wave_id = threadIdx.x >> 6;
  const int row_id = lane_id & 15;
  const int b_k_group = lane_id >> 4;

  const int k64_tile_count = size_k / kKPerBPayloadTile;
  const int n32_tile_count = size_n / kN32Cols;
  const int m_tiles_per_expert = ceil_div_int(size_m, kBlockM);
  const int64_t b_expert_stride = static_cast<int64_t>(size_n) * size_k;

  int task_id = blockIdx.x;
  const int task_stride = gridDim.x;
  int expert_task_begin = 0;

  for (int expert = 0; expert < experts; ++expert) {
    int valid_m = masked_m[expert];
    valid_m = valid_m < 0 ? 0 : valid_m;
    valid_m = valid_m > size_m ? size_m : valid_m;
    const int m_tile_count = ceil_div_int(valid_m, kBlockM);
    const int expert_task_count = m_tile_count * n_tiles;
    const int expert_task_end = expert_task_begin + expert_task_count;
    const uintx4 packed_b_resource = make_raw_buffer_resource(
        packed_b + static_cast<int64_t>(expert) * b_expert_stride);

    while (task_id >= expert_task_begin && task_id < expert_task_end) {
      const int expert_local_task = task_id - expert_task_begin;
      const int m_tile_id = expert_local_task / n_tiles;
      const int n_tile_id = expert_local_task - m_tile_id * n_tiles;
      const int m_block_start = m_tile_id * kBlockM;
      const int n_block_base = n_tile_id * kBlockN;
      const int64_t a_expert_offset = static_cast<int64_t>(expert) * size_m * size_k;

      AccPairFragment acc[kMSubtiles][kNPairsPerWave];
#pragma unroll
      for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
#pragma unroll
        for (int n_pair = 0; n_pair < kNPairsPerWave; ++n_pair) {
          acc[m_loop][n_pair] = zero_acc_pair_fragment();
        }
      }

      if (k64_tile_count > 0) {
        load_a_tile_to_lds_4w<kBlockM, kBlockK>(
            a_lds,
            a,
            a_expert_offset,
            m_block_start,
            0,
            valid_m,
            size_m,
            size_k,
            wave_id,
            lane_id,
            true);
        wait_vmcnt_barrier();
      }

      int read_stage = 0;
      int prefetch_stage = 1;

      BPairFragment b_k64_first[kNPairsPerWave];
      BPairFragment b_k64_second[kNPairsPerWave];
      BPairFragment b_next_k64_first[kNPairsPerWave];
      BPairFragment b_next_k64_second[kNPairsPerWave];
      Fp8x16Fragment a_k64_first[kMSubtiles];
      Fp8x16Fragment a_k64_second[kMSubtiles];

      if (k64_tile_count > 0) {
        load_a_fragments_for_k64_tile<kMSubtiles>(
            a_k64_first, a_lds, 0, row_id, b_k_group);
        load_a_fragments_for_k64_tile<kMSubtiles>(
            a_k64_second, a_lds, kKPerBPayloadTile, row_id, b_k_group);
      }

      if (size_k > 0) {
        load_b_pairs_2k64_precompute_offsets<kNPairsPerWave>(
            b_k64_first,
            b_k64_second,
            packed_b_resource,
            0,
            n32_tile_count,
            n_tile_id,
            wave_id,
            b_k_group,
            row_id);
        wait_vmcnt_zero();
      }

      for (int k_base_outer = 0; k_base_outer < size_k;
           k_base_outer += kBlockK) {
        const int k64_tile_base = k_base_outer / kKPerBPayloadTile;
        const int next_k_base = k_base_outer + kBlockK;
        const bool has_next_k = next_k_base < size_k;
        int8_t* a_lds_prefetch = a_lds + prefetch_stage * kAStageBytes;

        if (has_next_k) {
          load_a_tile_to_lds_4w<kBlockM, kBlockK>(
              a_lds_prefetch,
              a,
              a_expert_offset,
              m_block_start,
              next_k_base,
              valid_m,
              size_m,
              size_k,
              wave_id,
              lane_id,
              true);
          load_b_pairs_2k64_precompute_offsets<kNPairsPerWave>(
              b_next_k64_first,
              b_next_k64_second,
              packed_b_resource,
              k64_tile_base + 2,
              n32_tile_count,
              n_tile_id,
              wave_id,
              b_k_group,
              row_id);
        }

        wait_lgkmcnt_zero();

        accumulate_one_k64_tile_from_a_fragments<kMSubtiles, kNPairsPerWave>(
            a_k64_first,
            m_block_start,
            valid_m,
            b_k64_first,
            acc);

        accumulate_one_k64_tile_from_a_fragments<kMSubtiles, kNPairsPerWave>(
            a_k64_second,
            m_block_start,
            valid_m,
            b_k64_second,
            acc);

        if (has_next_k) {
          wait_vmcnt_barrier();
          load_a_fragments_for_k64_tile<kMSubtiles>(
              a_k64_first, a_lds_prefetch, 0, row_id, b_k_group);
          load_a_fragments_for_k64_tile<kMSubtiles>(
              a_k64_second, a_lds_prefetch, kKPerBPayloadTile, row_id,
              b_k_group);
#pragma unroll
          for (int n_pair = 0; n_pair < kNPairsPerWave; ++n_pair) {
            b_k64_first[n_pair] = b_next_k64_first[n_pair];
            b_k64_second[n_pair] = b_next_k64_second[n_pair];
          }
          read_stage = 1 - read_stage;
          prefetch_stage = 1 - prefetch_stage;
        }
      }

      barrier();
      stage_output_scales_to_lds(
          scale_tile_lds,
          a_scale,
          b_scale,
          make_raw_buffer_resource(a_scale),
          make_raw_buffer_resource(b_scale),
          expert,
          m_block_start,
          n_block_base,
          valid_m,
          size_m,
          size_n,
          threadIdx.x);
      wait_vmcnt_lgkmcnt_barrier();

      const int valid_rows_in_tile = valid_m - m_block_start;
      const int clamped_rows_in_tile =
          valid_rows_in_tile <= 0
              ? 0
              : (valid_rows_in_tile >= kBlockM ? kBlockM : valid_rows_in_tile);
      const int full_m_subtiles = clamped_rows_in_tile >> 4;
      const int tail_rows = clamped_rows_in_tile - full_m_subtiles * kMmacM;

      float a_scale_frag[kMSubtiles];
      int64_t c_offset_frag[kMSubtiles];
      const int64_t expert_c_row_base =
          static_cast<int64_t>(expert) * size_m + m_block_start;
#pragma unroll
      for (int m_loop = 0; m_loop < kMSubtiles; ++m_loop) {
        const int local_row = m_loop * kMmacM + row_id;
        a_scale_frag[m_loop] = a_scale_tile_lds[local_row];
        c_offset_frag[m_loop] =
            (expert_c_row_base + local_row) * static_cast<int64_t>(size_n);
      }

#pragma unroll
      for (int n_pair = 0; n_pair < kNPairsPerWave; ++n_pair) {
        const int wave_n32_index = n_pair * kNumWarps + wave_id;
        const int store_col_base =
            n_block_base + wave_n32_index * kN32Cols + b_k_group * 8;
        float b_scale_frag[8];
        const int local_scale_col_base = wave_n32_index * kN32Cols + b_k_group * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          b_scale_frag[i] = b_scale_tile_lds[local_scale_col_base + i];
        }

        switch (full_m_subtiles) {
          case 4:
            store_full_m_subtile_pair<3>(
                acc, n_pair, b_scale_frag, a_scale_frag, c_offset_frag, c,
                store_col_base);
          case 3:
            store_full_m_subtile_pair<2>(
                acc, n_pair, b_scale_frag, a_scale_frag, c_offset_frag, c,
                store_col_base);
          case 2:
            store_full_m_subtile_pair<1>(
                acc, n_pair, b_scale_frag, a_scale_frag, c_offset_frag, c,
                store_col_base);
          case 1:
            store_full_m_subtile_pair<0>(
                acc, n_pair, b_scale_frag, a_scale_frag, c_offset_frag, c,
                store_col_base);
          default:
            break;
        }

        if (tail_rows > 0) {
          switch (full_m_subtiles) {
            case 0:
              store_tail_m_subtile_pair<0>(
                  acc, n_pair, b_scale_frag, a_scale_frag, c_offset_frag, c,
                  store_col_base, row_id, tail_rows);
              break;
            case 1:
              store_tail_m_subtile_pair<1>(
                  acc, n_pair, b_scale_frag, a_scale_frag, c_offset_frag, c,
                  store_col_base, row_id, tail_rows);
              break;
            case 2:
              store_tail_m_subtile_pair<2>(
                  acc, n_pair, b_scale_frag, a_scale_frag, c_offset_frag, c,
                  store_col_base, row_id, tail_rows);
              break;
            case 3:
              store_tail_m_subtile_pair<3>(
                  acc, n_pair, b_scale_frag, a_scale_frag, c_offset_frag, c,
                  store_col_base, row_id, tail_rows);
              break;
            default:
              break;
          }
        }
      }

      if (enable_overlap) {
        // Drain each thread's outstanding C stores to the coherence point
        // before the block-wide barrier, so thread 0's signal update is
        // ordered after every thread's write-back of this (block_m, block_n).
        wait_vmcnt_zero();
      }
      barrier();
      if (enable_overlap && threadIdx.x == 0) {
        atomicAdd(signal + expert * m_tiles_per_expert + m_tile_id, 1);
      }
      task_id += task_stride;
    }

    expert_task_begin = expert_task_end;
  }
}

void launch(torch::Tensor a,
            torch::Tensor packed_b,
            torch::Tensor a_scale,
            torch::Tensor b_scale,
            torch::Tensor masked_m,
            bool enable_overlap,
            torch::Tensor signal,
            torch::Tensor c) {
  TORCH_CHECK(a.is_cuda() && packed_b.is_cuda() && a_scale.is_cuda() && b_scale.is_cuda() && masked_m.is_cuda() && c.is_cuda(), "all tensors must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn, "a must be float8_e4m3fn");
  TORCH_CHECK(packed_b.scalar_type() == at::kFloat8_e4m3fn, "packed_b must be float8_e4m3fn");
  TORCH_CHECK(a_scale.scalar_type() == at::kFloat, "a_scale must be float32");
  TORCH_CHECK(b_scale.scalar_type() == at::kFloat, "b_scale must be float32");
  TORCH_CHECK(masked_m.scalar_type() == at::kInt, "masked_m must be int32");
  TORCH_CHECK(c.scalar_type() == at::kBFloat16, "c must be bf16");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(packed_b.is_contiguous(), "packed_b must be contiguous");
  TORCH_CHECK(a_scale.is_contiguous(), "a_scale must be contiguous");
  TORCH_CHECK(b_scale.is_contiguous(), "b_scale must be contiguous");
  TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");
  TORCH_CHECK(c.is_contiguous(), "c must be contiguous");
  TORCH_CHECK(a.dim() == 3, "a must be [E,M,K]");
  TORCH_CHECK(packed_b.dim() == 7, "packed_b must be [E,K/64,N/32,2,4,16,16]");
  TORCH_CHECK(a_scale.dim() == 2 || (a_scale.dim() == 3 && a_scale.size(-1) == 1), "a_scale must be [E,M] or [E,M,1]");
  TORCH_CHECK(b_scale.dim() == 2 || (b_scale.dim() == 3 && b_scale.size(-1) == 1), "b_scale must be [E,N] or [E,N,1]");
  TORCH_CHECK(c.dim() == 3, "c must be [E,M,N]");
  TORCH_CHECK(masked_m.dim() == 1, "masked_m must be [E]");

  const int experts = a.size(0);
  const int size_m = a.size(1);
  const int size_k = a.size(2);
  const int size_n = c.size(2);
  TORCH_CHECK(size_k % kBlockK == 0, "K must be divisible by 128");
  TORCH_CHECK(size_n % kBlockN == 0, "N must be divisible by 256");
  TORCH_CHECK(static_cast<int64_t>(size_n) * size_k <= (int64_t{1} << 31),
              "per-expert packed weight must fit within a 2 GiB buffer resource");
  TORCH_CHECK(packed_b.size(0) == experts, "packed_b expert dim mismatch");
  TORCH_CHECK(packed_b.size(1) == size_k / 64, "packed_b K dim mismatch");
  TORCH_CHECK(packed_b.size(2) == size_n / 32, "packed_b N32 dim mismatch");
  TORCH_CHECK(packed_b.size(3) == 2 && packed_b.size(4) == 4 &&
                  packed_b.size(5) == 16 && packed_b.size(6) == 16,
              "packed_b tail dims mismatch");
  TORCH_CHECK(masked_m.numel() == experts, "masked_m shape mismatch");
  TORCH_CHECK(a_scale.size(0) == experts && a_scale.size(1) == size_m, "a_scale shape mismatch");
  TORCH_CHECK(b_scale.size(0) == experts && b_scale.size(1) == size_n, "b_scale shape mismatch");
  TORCH_CHECK(c.size(0) == experts && c.size(1) == size_m, "c shape mismatch");

  int32_t* signal_ptr = nullptr;
  if (enable_overlap) {
    TORCH_CHECK(signal.is_cuda() && signal.scalar_type() == at::kInt &&
                    signal.is_contiguous(),
                "signal must be a contiguous CUDA int32 tensor");
    TORCH_CHECK(signal.numel() >=
                    static_cast<int64_t>(experts) *
                        ((size_m + kBlockM - 1) / kBlockM),
                "signal must have at least E * ceil(M/", kBlockM, ") elements");
    signal_ptr = reinterpret_cast<int32_t*>(signal.data_ptr());
  }

  c10::hip::HIPGuardMasqueradingAsCUDA guard(a.device());
  int device = 0;
  hipGetDevice(&device);
  hipDeviceProp_t_v2 prop;
  hipGetDeviceProperties_v2(&prop, device);

  int blocks_per_cu = 3;

  const int n_tiles = size_n / kBlockN;
  const int grid_x = prop.multiProcessorCount * blocks_per_cu;
  const dim3 block(kThreads, 1, 1);
  const dim3 grid(grid_x, 1, 1);
  const hipStream_t stream = at::cuda::getCurrentHIPStream();
 hipLaunchKernelGGL(( grouped_w8a8_fp8_nt_masked_bm64_bn256_bk128_pair32_persistent_a_double_buffer_writeback16_kernel), dim3(grid), dim3(block), kSharedBytes, stream,
      reinterpret_cast<const int8_t*>(a.data_ptr()),
      reinterpret_cast<const int8_t*>(packed_b.data_ptr()),
      reinterpret_cast<const float*>(a_scale.data_ptr()),
      reinterpret_cast<const float*>(b_scale.data_ptr()),
      reinterpret_cast<const int32_t*>(masked_m.data_ptr()),
      reinterpret_cast<bf16*>(c.data_ptr()),
      experts,
      size_m,
      size_n,
      size_k,
      n_tiles,
      enable_overlap,
      signal_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace w8a8_fp8_masked_bm64_bk128
