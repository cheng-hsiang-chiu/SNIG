#pragma once

#include <Eigen/Core>
#include <taskflow/taskflow.hpp>
#include <taskflow/syclflow.hpp>
#include <SNIG/utility/reader.hpp>
#include <SNIG/utility/matrix_format.h>
//#include <SNIG/utility/cuda_error.hpp>
#include <SNIG/snig/kernel.hpp>
#include <SNIG/utility/scoring.hpp>
#include <SNIG/base/base.hpp>
#include <vector>



namespace std {
  namespace fs = experimental::filesystem;  
}

namespace snig{

class mxm_kernel;

template <typename T>
class SNIG : public Base<T> {

  static_assert(
    std::is_same<T, float>::value || std::is_same<T, double>::value,
    "data type must be either float or double");
  
  private:
    
    size_t _batch_size;

    size_t _num_weight_buffers;
    
    T* _source_Y;
    
    bool* _source_is_nonzero_row;
    
    std::vector<std::vector<T*> > _dev_Y;
    
    std::vector<std::vector<bool*> > _dev_is_nonzero_row;
    
    std::vector<std::vector<int*> > _dev_W;

    size_t _batch_ylen;
    
    size_t _batch_ysize;
    
    int* _results;

    void _set_parameters(
      const size_t num_inputs,
      const size_t batch_size,
      const size_t num_weight_buffers,
      const size_t num_gpus);

    void _preprocess(const std::fs::path& input_path);
  
    void  _infer();

    void _input_alloc();

    void _weight_alloc();

    void _result_alloc();

  public:

    //sycl::queue queue;
  
    /*
    SNIG(
      const dim3& threads,
      const std::fs::path& weight_path,
      const T bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120);
    */
     
    SNIG(
      const std::string& weight_path, //const std::fs::path& weight_path,
      const T bias = -.3f,
      const size_t num_neurons_per_layer = 1024,
      const size_t num_layers = 120);
    
    ~SNIG();

    Eigen::Matrix<int, Eigen::Dynamic, 1> infer(
      const std::fs::path& input_path,
      const size_t num_inputs,
      const size_t batch_size,
      const size_t num_buff,
      const size_t num_gpus);

};

// ----------------------------------------------------------------------------
// Definition of SNIG
// ----------------------------------------------------------------------------
/*
template <typename T>
SNIG<T>::SNIG(
  const dim3& threads,
  const std::fs::path& weight_path,
  const T bias,
  const size_t num_neurons_per_layer,
  const size_t num_layers) :
  Base<T>(threads, weight_path, bias, num_neurons_per_layer, num_layers) {
    Base<T>::log("Constructing SNIG engine......", "\n");
  }
*/


template <typename T>
SNIG<T>::SNIG(
  const std::string& weight_path,  //const std::fs::path& weight_path,
  const T bias,
  const size_t num_neurons_per_layer,
  const size_t num_layers) :
  Base<T>(weight_path, bias, num_neurons_per_layer, num_layers) {
    Base<T>::log("Constructing SNIG engine......", "\n");
    std::cout << "SNIG constructor\n";
  }



template <typename T>
SNIG<T>::~SNIG() {
 /* 
  sycl::free(_source_Y, Base<T>::_queue);
  sycl::free(_source_is_nonzero_row, Base<T>::_queue);

  for (auto& W_in_dev : _dev_W) {
    for (auto& each_W : W_in_dev) {
      sycl::free(each_W, Base<T>::_queue);
    }
  }

  for (auto& Y_in_dev : _dev_Y) {
      sycl::free(Y_in_dev[1], Base<T>::_queue);
  }
  
  for (auto& rowsY_in_dev : _dev_is_nonzero_row) {
      sycl::free(rowsY_in_dev[1], Base<T>::_queue);
  }

  sycl::free(_results, Base<T>::_queue);
  */
}



template <typename T>
Eigen::Matrix<int, Eigen::Dynamic, 1> SNIG<T>::infer(
  const std::fs::path& input_path,
  const size_t num_inputs,
  const size_t batch_size,
  const size_t num_weight_buffers,
  const size_t num_gpus) {
    
  Base<T>::log("Using ", num_gpus, " GPUs", "\n");
  Base<T>::log("Total input size : ", num_inputs, "\n");
  Base<T>::log("Input batch size : ", batch_size, "\n");
  Base<T>::log("Number of weight buffers : ", num_weight_buffers, "\n\n");
  
  _set_parameters(
    num_inputs,
    batch_size,
    num_weight_buffers,
    num_gpus);
  
  _preprocess(input_path);
  
  _infer();
  
  return arr_to_Eigen_int(_results, Base<T>::_num_inputs);
  
  //Eigen::Matrix<int, Eigen::Dynamic, 1> result;
  //return result;
}



template <typename T>
void SNIG<T>::_set_parameters(
  const size_t num_inputs,
  const size_t batch_size,
  const size_t num_weight_buffers,
  const size_t num_gpus) {

  Base<T>::_num_inputs = num_inputs;
  Base<T>::_num_gpus = num_gpus;
  _num_weight_buffers = num_weight_buffers;

  _batch_size = batch_size;
  _batch_ylen = _batch_size * Base<T>::_num_neurons;
  _batch_ysize = _batch_ylen * sizeof(T);

  _dev_W.reserve(Base<T>::_num_gpus);
  _dev_Y.reserve(Base<T>::_num_gpus);
  _dev_is_nonzero_row.reserve(Base<T>::_num_gpus);
  /*
  std::cout << "Base<T>::_num_inputs = " << Base<T>::_num_inputs << '\n';
  std::cout << "Base<T>::_num_gpus = " << Base<T>::_num_gpus << '\n';
  std::cout << "_num_weight_buffers = " << num_weight_buffers << '\n';
  std::cout << "_batch_size = " << _batch_size << '\n';
  std::cout << "_batch_ylen = " << _batch_ylen << '\n';
  std::cout << "_batch_ysize = " << _batch_ysize << '\n';
  */
}



template <typename T>
void SNIG<T>::_preprocess(const std::fs::path& input_path) {
  Base<T>::log("Preprocessing...... ");
  Base<T>::tic();

  //weight allocation
  _weight_alloc();
  //input allocation
  _input_alloc();
  //final results allocation
  _result_alloc();
  
  //read input
  read_input_binary<T>(input_path, _source_Y);

  Base<T>::toc();
  Base<T>::log("Finish preprocessing with ", Base<T>::duration(), " ms", "\n");
}





template <typename T>
void SNIG<T>::_infer() {
  Base<T>::log("Start inference...... ", "\n");
  Base<T>::tic();
  
  //Use taskflow and syclGraph to implement task graph
  tf::Taskflow taskflow("SNIG");
  tf::Executor executor;
  
  std::vector<tf::Task> first_fetchs;
  std::vector<tf::Task> syclflows;
  std::vector<tf::Task> fetchs;
  
  first_fetchs.reserve(Base<T>::_num_gpus);
  syclflows.reserve(Base<T>::_num_gpus);
  fetchs.reserve(Base<T>::_num_gpus);
  
  std::atomic<size_t> finished_inputs{0};
  std::vector<int*> dev_results(Base<T>::_num_gpus, nullptr);
  
  //dim3 grid_dim(_batch_size, Base<T>::_num_secs, 1);
  //std::cout << "_num_secs = " << Base<T>::_num_secs << '\n';
  tf::Task start = taskflow.emplace([](){}).name("start");
  
  for (size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    first_fetchs.emplace_back(taskflow.emplace([&, dev](){
      int is_end = 1;
      size_t beg_inputs = finished_inputs.fetch_add(_batch_size);
      
      if (beg_inputs < Base<T>::_num_inputs) {
        _dev_Y[dev][0] = _source_Y + beg_inputs * Base<T>::_num_neurons;
        _dev_is_nonzero_row[dev][0] = _source_is_nonzero_row + beg_inputs * Base<T>::_num_secs;
        dev_results[dev] = _results + beg_inputs;

        //cudaMemPrefetchAsync(_dev_Y[dev][0], _batch_ysize, dev, NULL);
        //cudaMemPrefetchAsync(_dev_is_nonzero_row[dev][0], sizeof(bool) * _batch_size * Base<T>::_num_secs, dev, NULL);
        //cudaMemPrefetchAsync(dev_results[dev], sizeof(int) * _batch_size, dev, NULL); 
        is_end = 0;
      }
      return is_end;
    }).name("first_fetch"));
  
    
    syclflows.emplace_back(taskflow.emplace_on([&, dev](tf::syclFlow& sf){
      std::vector<tf::syclTask> weight_copies;
      std::vector<tf::syclTask> infers;
      
      weight_copies.reserve(Base<T>::_num_layers);
      infers.reserve(Base<T>::_num_layers);
      
      for (size_t cur_layer = 0; cur_layer < Base<T>::_num_layers; cur_layer += _num_weight_buffers) {
        for (size_t k = 0; k < _num_weight_buffers; ++k) {
          //tasks of syclflow
          
          weight_copies.emplace_back(sf.copy(
            _dev_W[dev][k],
            Base<T>::_host_pinned_weight + (cur_layer + k) * Base<T>::_pp_wlen,
            Base<T>::_pp_wlen
          ).name("weight_copy"));
         
          // transformed CSC weight matrix equals to CSR with exchanged row and col
          int* col_w = _dev_W[dev][k];
          int* row_w = _dev_W[dev][k] + Base<T>::_num_neurons * Base<T>::_num_secs + 1;
          T* val_w = (T*)(_dev_W[dev][k] + Base<T>::_p_w_index_len);
           
          auto M = _batch_size * 2;
          auto K = 512 * Base<T>::_num_secs;

          auto resized_batch = (M % 2 == 0) ? M : (M + 2 - M % 2);
          auto resized_secs = (K % 512 == 0) ? K : (K + 512 - K % 512); 
           
          //T* M_result;
          //bool* M_isnonzero;

          //const sycl::property_list props = {sycl::property::buffer::use_host_ptr()}; 
          //sycl::buffer<T> b_result(M_result, sycl::range<1>{Base<T>::_sec_size});
          //sycl::buffer<bool> b_isnonzero(M_isnonzero, sycl::range<1>{2});
           
          infers.emplace_back(sf.on(
            [=](sycl::handler& cgh) { 
              //auto p_result = b_result.get_access<sycl::access::mode::read_write>(cgh);
              //auto p_isnonzero = b_isnonzero.get_access<sycl::access::mode::read_write>(cgh);
              //auto localRange = sycl::range<1>(2);
               
              sycl::accessor<T, 1, 
                sycl::access::mode::read_write, 
                sycl::access::target::local> p_b_result(sycl::range<1>{Base<T>::_sec_size}, cgh);

              sycl::accessor<bool, 1, 
                sycl::access::mode::read_write, 
                sycl::access::target::local> p_b_is_nonzero(sycl::range<1>{2}, cgh);
                
              auto si_dev_Y = _dev_Y[dev][k % 2];
              auto si_dev_is_nonzero_row = _dev_is_nonzero_row[dev][k % 2];
              auto si_dev_Y_1 = _dev_Y[dev][(k + 1) % 2];
              auto si_dev_is_nonzero_row_1 = _dev_is_nonzero_row[dev][(k + 1) % 2];
              auto si_sec_size = Base<T>::_sec_size;
              auto si_num_secs = Base<T>::_num_secs;
              auto si_num_neurons = Base<T>::_num_neurons;
              auto si_bias = Base<T>::_bias;
               
              cgh.parallel_for<mxm_kernel>(sycl::nd_range<2>{
                sycl::range<2>(resized_secs, resized_batch), 
                sycl::range<2>(512, 2)},
                [=](sycl::nd_item<2> item) {
                  //snig_inference1<T>(item);
                   
                  snig_inference<T>(
                    si_dev_Y,
                    si_dev_is_nonzero_row,
                    si_sec_size,
                    si_num_secs,
                    si_num_neurons,
                    col_w,
                    row_w,
                    val_w,
                    si_bias,
                    si_dev_is_nonzero_row_1,
                    si_dev_Y_1,
                    item,
                    p_b_result,
                    p_b_is_nonzero
                  );
                }
              );
            }).name("Inference")
          );
        }
      }
      

      
      // TODO: consider parameterizing the thread numbers
      //tf::syclTask ident = cf.kernel(16, 512, 0, identify<T>, _dev_Y[dev][0], _batch_size, Base<T>::_num_neurons, dev_results[dev]);
       
      auto bsize = _batch_size;
      auto nns = Base<T>::_num_neurons;
      auto dY = _dev_Y[dev][0];
      auto dr = dev_results[dev];
      
      std::cout << "batch size = " << bsize << '\n';
      
      tf::syclTask ident = sf.parallel_for(
        sycl::nd_range<1>{sycl::range<1>(8192), sycl::range<1>(512)}, 
        [=](sycl::nd_item<1> item) {

          //int tid = item.get_global_linear_id();
          int tid = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
          //auto reduction = [](T a, T b){ return a + b; };
          for (int i = tid; i < bsize; i += item.get_group_range(0) * item.get_local_range(0)) {
            T sum;
            for (auto index = i * nns; index < (i+1) * nns; ++index) {
              sum = sum + *(dY+index);
            }
            *(dr+i) = sum > 0 ? 1 : 0;
            item.barrier(sycl::access::fence_space::local_space);
          }

          //identify<T>(dY, bsize, nns, dr, item);                                 
        }
      ).name("ident");
      
       
      //dependencies of syclflow
      for (size_t cur_layer = 0; cur_layer < Base<T>::_num_layers; ++cur_layer) {
        weight_copies[cur_layer].precede(infers[cur_layer]);
        
        if (cur_layer + _num_weight_buffers < Base<T>::_num_layers) {
          infers[cur_layer].precede(weight_copies[cur_layer + _num_weight_buffers]);
        }

        if (cur_layer + 1 < Base<T>::_num_layers) {
          infers[cur_layer].precede(infers[cur_layer + 1]);
        }
      }
      
      infers[Base<T>::_num_layers - 1].precede(ident);
      
    }, Base<T>::queue).name("GPU"));

     
    fetchs.emplace_back(taskflow.emplace([&, dev](){
      int is_end = 1;
      size_t beg_inputs = finished_inputs.fetch_add(_batch_size);
      
      if (beg_inputs < Base<T>::_num_inputs) {
        _dev_Y[dev][0] = _source_Y + beg_inputs * Base<T>::_num_neurons;
        _dev_is_nonzero_row[dev][0] = _source_is_nonzero_row + beg_inputs * Base<T>::_num_secs;
        dev_results[dev] = _results + beg_inputs;
        
        //cudaMemPrefetchAsync(_dev_Y[dev][0], _batch_ysize, dev, NULL);
        //cudaMemPrefetchAsync(_dev_is_nonzero_row[dev][0], sizeof(bool) * _batch_size * Base<T>::_num_secs, dev, NULL);
        //cudaMemPrefetchAsync(dev_results[dev], sizeof(int) * _batch_size, dev, NULL);
        
        is_end = 0;
      }
      return is_end;
    }).name("fetch"));
  }

  tf::Task stop = taskflow.emplace([](){ std::cout << "stop inference ...\n";}).name("stop");
  
  //dependencies of taskflow
  for (size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    start.precede(first_fetchs[dev]);
    first_fetchs[dev].precede(syclflows[dev], stop);
    syclflows[dev].precede(fetchs[dev]);
    fetchs[dev].precede(syclflows[dev], stop);

    //first_fetchs[dev].precede(syclflows[dev],stop);
    //syclflows[dev].precede(stop);
  }
  
  executor.run(taskflow).wait();

  //taskflow.dump(std::cout);

  //checkCuda(cudaSetDevice(0));
  
  Base<T>::toc();
  Base<T>::log("Finish inference with ", Base<T>::duration(), " ms", "\n");
}



template <typename T>
void SNIG<T>::_weight_alloc() {
  std::vector<int*> W(_num_weight_buffers, nullptr);

  for (size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    for (auto& each_W : W) {
      each_W = sycl::malloc_device<int>(
        Base<T>::_pp_wsize / sizeof(int), 
        Base<T>::queue
      );
    }
    _dev_W.push_back(W);
  }
}



template <typename T>
void SNIG<T>::_input_alloc() {
  size_t ylen = Base<T>::_num_inputs * Base<T>::_num_neurons;
  size_t ysize = ylen * sizeof(T);

  _source_Y = sycl::malloc_shared<T>(
    ysize / sizeof(T), 
    Base<T>::queue
  );
   
  _source_is_nonzero_row = sycl::malloc_shared<bool>(
    Base<T>::_num_inputs * Base<T>::_num_secs, 
    Base<T>::queue
  );
   
  Base<T>::queue.memset(
    _source_is_nonzero_row, 
    1, 
    sizeof(bool) * Base<T>::_num_inputs * Base<T>::_num_secs
  ).wait();

  std::vector<T*> Y{2, nullptr};
  std::vector<bool*> is_nonzero_row{2, nullptr};
  
  for (size_t dev = 0; dev < Base<T>::_num_gpus; ++dev) {
    Y[1] = sycl::malloc_device<T>(
      _batch_ysize / sizeof(T), 
      Base<T>::queue
    );

    is_nonzero_row[1] = sycl::malloc_device<bool>(
      _batch_size * Base<T>::_num_secs, 
      Base<T>::queue
    );
    
    Base<T>::queue.memset(Y[1], 0, _batch_ysize).wait();
    Base<T>::queue.memset(
      is_nonzero_row[1], 
      0, 
      sizeof(bool) * _batch_size * Base<T>::_num_secs
    ).wait();

    _dev_Y.push_back(Y);
    _dev_is_nonzero_row.push_back(is_nonzero_row);
  }
  /*
  std::cout << "Base<T>::_num_inputs = " << Base<T>::_num_inputs << '\n';
  std::cout << "Base<T>::_num_secs = " << Base<T>::_num_secs << '\n';
  std::cout << "_batch_size = " << _batch_size << '\n';
  std::cout << "_batch_ysize = " << _batch_ysize << '\n';
  */
}



template <typename T>
void SNIG<T>::_result_alloc() {
  _results = sycl::malloc_shared<int>(
    Base<T>::_num_inputs, 
    Base<T>::queue
  );

  Base<T>::queue.memset(
    _results, 
    0, 
    sizeof(int) * Base<T>::_num_inputs
  ).wait();
}




}// end of namespace snig ----------------------------------------------
