#pragma once

namespace snig{

/*
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
  T* Y_1,
  sycl::nd_item<2> item,
  sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local> p_b_results,
  sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local> p_b_is_nonzero
);

template <typename T>
void snig_inference1(
  sycl::nd_item<2> item
);
*/


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
  T* Y_1,
  sycl::nd_item<2> item,
  const sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>& p_b_results, 
  const sycl::accessor<bool, 1, sycl::access::mode::read_write, sycl::access::target::local>& p_b_is_nonzero
);
//-----------------------------------------------------------------------------
//Definition of kernel function
//-----------------------------------------------------------------------------

template <typename T>
void snig_inference1(
  sycl::nd_item<2> item) {
  int tid = item.get_global_linear_id();
}



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
  T* Y_1,
  sycl::nd_item<2> item,
  const sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>& p_b_results,
  const sycl::accessor<bool, 1, sycl::access::mode::read_write, sycl::access::target::local>& p_b_is_nonzero) {

  /*  
  auto localRange = sycl::range<1>(16 * 16);
  sycl::accessor<T, 1, 
    sycl::access::mode::read_write, 
    sycl::access::target::local> p_b_result(localRange, cgh);
 
  sycl::accessor<T, 1, 
    sycl::access::mode::read_write, 
    sycl::access::target::local> p_b_isnonzero(localRange, cgh);
  */
  
  int tid = item.get_local_id(0) * 2 + item.get_local_id(1);
  ////int tid = threadIdx.y * blockDim.x + threadIdx.x;
  //r = blockIdx.x
  //s_o = blockIdx.y
  ////int num_threads = blockDim.x * blockDim.y;
  int num_threads = 1024;
  
  //num_secs is small enough to compute by each single thread
  bool is_all_zero = true;
  for (size_t s_i = 0; s_i < num_secs; ++s_i) {
    is_all_zero &= !is_nonzero_row_0[item.get_group(1) * num_secs + s_i];
    ////is_all_zero &= !is_nonzero_row_0[blockIdx.x * num_secs + s_i];
  }
  
  if (is_all_zero) {
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
    if (is_nonzero_row_1[item.get_group(1) * num_secs + item.get_group(0)]) {
      for (size_t j = tid; j < sec_size; j += num_threads) {
        ////Y_1[blockIdx.x * num_neurons + blockIdx.y * sec_size + j] = 0;
        Y_1[item.get_group(1) * num_neurons + item.get_group(0) * sec_size + j] = 0;
      }
      item.barrier(sycl::access::fence_space::local_space);
      ////__syncthreads();
      
      if (tid == 0) {
        is_nonzero_row_1[item.get_group(1) * num_secs + item.get_group(0)] = false;
      } 
    }
    return;
  }
     
  //forward feeding
  ////extern __shared__ T results[];

  //set results to bias directly
  for (size_t k = tid; k < sec_size; k += num_threads) {
    p_b_results[k] = bias;  
  }
   
  //use bool array size of 2 (is_nonzero) in share memory to avoid synchronization
  //is_nonzero[1] represents whether this row is nonzero
  //if is_nonzero[1] is true, this row is nonzero
  ////__shared__ bool is_nonzero[2];

  if (tid == 0) {
    p_b_is_nonzero[1] = false;
  }
  ////__syncthreads();
  item.barrier(sycl::access::fence_space::local_space);
   
  for (size_t s_i = 0; s_i < num_secs; ++s_i) {
    if (!is_nonzero_row_0[item.get_group(1) * num_secs + s_i]) {
      continue;
    }
     
    for (size_t j = item.get_local_id(0) + s_i * sec_size; 
         j < (s_i + 1) * sec_size; 
         j += 512) {

      T valY = Y_0[item.get_group(1) * num_neurons + j];
      if (valY == 0) {
        continue;
      }
      
      int beg_w = col_w[item.get_group(0) * num_neurons + j] + item.get_local_id(1);
      int end_w = col_w[item.get_group(0) * num_neurons + j + 1];
      
      for (int k = beg_w; k < end_w; k += 2) {
        int roww = row_w[k];
        T valw = val_w[k];
        if ((roww - item.get_group(0) * sec_size) >= 0 && 
            (roww - item.get_group(0) * sec_size) < sec_size) { 
         
          auto ref = sycl::ONEAPI::atomic_ref<
            T,
            sycl::ONEAPI::memory_order_seq_cst,
            sycl::ONEAPI::memory_scope::device,
            sycl::access::address_space::local_space
          >{p_b_results[roww - item.get_group(0) * sec_size]};
          
          ref.fetch_add(valY * valw);
        }
        else {
          continue;
        }
        //atomicAdd(&p_b_results[roww - item.get_group(0) * sec_size], valY * valw);
      }  
    }
  }
  item.barrier(sycl::access::fence_space::local_space);
   
  for (size_t i = tid; i < sec_size; i += num_threads) {
    T v = std::min(T(32), std::max(p_b_results[i], T(0)));
    Y_1[item.get_group(1) * num_neurons + item.get_group(0) * sec_size + i] = v;
    p_b_is_nonzero[v != 0] = true;
  }
  
  //if one thread sets is_nonzero[1] to true
  //meaning this row is nonzero
  //toggle is_nonzero_row_1[this row] to true
  ////__syncthreads();
  item.barrier(sycl::access::fence_space::local_space);
  if (tid == 0) {
    is_nonzero_row_1[item.get_group(1) * num_secs + item.get_group(0)] = p_b_is_nonzero[1];
  }
  
}




}// end of namespace snig ----------------------------------------------
