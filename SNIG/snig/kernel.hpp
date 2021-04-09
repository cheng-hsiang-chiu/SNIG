#pragma once

namespace snig{

template <typename T>
void snig_inference(
  const T* Y_0,
  const bool* is_nonzero_row_0,
  const size_t sec_size,
  const size_t num_sec,
  const size_t num_neurons,
  const int* col_w,
  const int* row_w,
  const T* val_w,
  const T bias,
  bool* is_nonzero_row_1,
  T* Y_1
);

//-----------------------------------------------------------------------------
//Definition of kernel function
//-----------------------------------------------------------------------------

template <typename T>
void snig_inference(
  const T* Y_0,
  const bool* is_nonzero_row_0,
  const size_t sec_size,
  const size_t num_secs,
  const size_t num_neurons,
  const int* col_w,
  const int* row_w,
  const T* val_w,
  const T bias,
  bool* is_nonzero_row_1,
  T* Y_1) {

  int row = item.get_global_id(0);
  int col = item.get_global_id(1);
  int tid = item.get_global_linear_id();
  ////int tid = threadIdx.y * blockDim.x + threadIdx.x;
  //r = blockIdx.x
  //s_o = blockIdx.y
  ////int num_threads = blockDim.x * blockDim.y;
  int num_threads = 256;

  //num_secs is small enough to compute by each single thread
  bool is_all_zero = true;
  for(size_t s_i = 0; s_i < num_secs; ++s_i) {
    is_all_zero &= !is_nonzero_row_0[item.get_group(0) * num_secs + s_i];
    ////is_all_zero &= !is_nonzero_row_0[blockIdx.x * num_secs + s_i];
  }

  if(is_all_zero) {
    //incremental memory resetting
    //avoid calling cudaMemset
    ////if(is_nonzero_row_1[blockIdx.x * num_secs + blockIdx.y]) {
    ////  for(size_t j = tid; j < sec_size; j += num_threads) {
    ////    Y_1[blockIdx.x * num_neurons + blockIdx.y * sec_size + j] = 0;
    ////  }
    ////  __syncthreads();
    ////  if(tid == 0) {
    ////    is_nonzero_row_1[blockIdx.x * num_secs + blockIdx.y] = false;
    ////  } 
    ////}
    if(is_nonzero_row_1[item.get_group(0) * num_secs + item.get_group(1)]) {
      for(size_t j = tid; j < sec_size; j += num_threads) {
        ////Y_1[blockIdx.x * num_neurons + blockIdx.y * sec_size + j] = 0;
        Y_1[item.get_group(0) * num_neurons + item.get_group(1) * sec_size + j] = 0;
      }
      ////__syncthreads();
      if(tid == 0) {
        is_nonzero_row_1[item.get_group(0) * num_secs + item.get_group(1)] = false;
      } 
    }
    return;
  }

  //forward feeding
  ////extern __shared__ T results[];
  T* results = malloc_device<T>(N, q);

  //set results to bias directly
  for(size_t k = tid; k < sec_size; k += num_threads) {
    results[k] = bias;  
  }

  //use bool array size of 2 (is_nonzero) in share memory to avoid synchronization
  //is_nonzero[1] represents whether this row is nonzero
  //if is_nonzero[1] is true, this row is nonzero
  ////__shared__ bool is_nonzero[2];
  bool* is_nonzero = malloc_device<bool>(2, q);

  if(tid == 0) {
    is_nonzero[1] = false;
  }
  ////__syncthreads();

  for(size_t s_i = 0; s_i < num_secs; ++s_i) {
    if(!is_nonzero_row_0[item.get_group(0) * num_secs + s_i]) {
      continue;
    }
    ////if(!is_nonzero_row_0[blockIdx.x * num_secs + s_i]) {
    ////  continue;
    ////}
    for(size_t j = item.get_global_id(1) + s_i * sec_size; j < (s_i + 1) * sec_size; j += 16) {
      T valY = Y_0[item.get_group(0) * num_neurons + j];
      if(valY == 0) {
        continue;
      }
      int beg_w = col_w[item.get_group(1) * num_neurons + j] + item.get_global_id(0);
      int end_w = col_w[item.get_group(1) * num_neurons + j + 1];
      for(int k = beg_w; k < end_w; k += 16) {
        int roww = row_w[k];
        T valw = val_w[k];
        atomicAdd(&results[roww - item.get_group(1) * sec_size], valY * valw);
      }
    }
    ////for(size_t j = threadIdx.y + s_i * sec_size; j < (s_i + 1) * sec_size; j += blockDim.y) {
    ////  T valY = Y_0[blockIdx.x * num_neurons + j];
    ////  if(valY == 0) {
    ////    continue;
    ////  }
    ////  int beg_w = col_w[blockIdx.y * num_neurons + j] + threadIdx.x;
    ////  int end_w = col_w[blockIdx.y * num_neurons + j + 1];
    ////  for(int k = beg_w; k < end_w; k += blockDim.x) {
    ////    int roww = row_w[k];
    ////    T valw = val_w[k];
    ////    atomicAdd(&results[roww - blockIdx.y * sec_size], valY * valw);
    ////  }
    ////}
  }
  ////__syncthreads();
  for(size_t i = tid; i < sec_size; i += num_threads) {
    T v = min(T(32), max(results[i], T(0)));
    Y_1[item.get_group(0) * num_neurons + item.get_group(1) * sec_size + i] = v;
    ////Y_1[blockIdx.x * num_neurons + blockIdx.y * sec_size + i] = v;
    is_nonzero[v != 0] = true;
  }

  //if one thread sets is_nonzero[1] to true
  //meaning this row is nonzero
  //toggle is_nonzero_row_1[this row] to true
  ////__syncthreads();
  if(tid == 0) {
    is_nonzero_row_1[item.get_group(0) * num_secs + item.get_group(1)] = is_nonzero[1];
    ////is_nonzero_row_1[blockIdx.x * num_secs + blockIdx.y] = is_nonzero[1];
  }
}

}// end of namespace snig ----------------------------------------------
