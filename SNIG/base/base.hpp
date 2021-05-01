#pragma once

#include <SNIG/utility/utility.hpp>
#include <chrono>
#include <taskflow/syclflow.hpp>
#include <string>


namespace snig {

template <typename T>
class Base {
  public:
    sycl::queue queue{sycl::gpu_selector{}};

  protected:
    //model configuration
    T _bias;
    size_t _num_neurons;
    size_t _num_layers;
    size_t _num_gpus;
    size_t _num_inputs;
    
    //Both SNIG and BF use maximum external shared memory
    //_num_secs == N_SLAB
    //_sec_size == COL_BLK
    size_t _num_secs;
    size_t _sec_size;

    //weights
    int* _host_pinned_weight;
    size_t _max_nnz;
    size_t _pad {0};
    size_t _p_w_index_len;
    size_t _pp_w_index_len;
    size_t _pp_wlen;
    size_t _pp_wsize;

    //sycl::queue queue;
    //kernel configuration
    //dim3 _threads{32, 32, 1};

    /*
    Base(
      const dim3& threads,
      const std::fs::path& weight_path,
      const T bias,
      const size_t num_neurons,
      const size_t num_layers
    );
    */
    
    Base(
      const std::fs::path& weight_path,
      const T bias,
      const size_t num_neurons,
      const size_t num_layers
    );
    
    //virtual ~Base();
    ~Base();
  
    //  API: cout("my ", string, " is ", a, b, '\n');
    //       -> cout << "my" << string << " is " << a << b << '\n';
    template <typename... ArgsT>
    void log(ArgsT&&... args) const;
    

    void tic();

    void toc();
    
    auto duration();

  private:

    std::chrono::time_point<std::chrono::steady_clock> _tic;
    std::chrono::time_point<std::chrono::steady_clock> _toc;
    bool _enable_counter{false};
    bool _enable_toc{false};

    void _load_weight(const std::fs::path& weight_path); 

    template <typename L>
    void _cout(L&& last) const;

    template <typename First, typename... Remain>
    void _cout(First&& item, Remain&&... remain) const;

    size_t num_neurons() const;

    size_t num_layers() const;

    virtual void _preprocess(const std::fs::path& input_path) = 0;
    
    virtual void _weight_alloc() = 0;

    virtual void _input_alloc() = 0;

    virtual void _result_alloc() = 0;

    //virtual void _infer() = 0;
};

// ----------------------------------------------------------------------------
// Definition of Base
// ----------------------------------------------------------------------------
/*
template <typename T>
Base<T>::Base(
  const dim3& threads,
  const std::fs::path& weight_path,
  const T bias,
  const size_t num_neurons,
  const size_t num_layers
) : 
  _bias{bias},
  _num_neurons{num_neurons},
  _num_layers{num_layers},
  //_threads{threads}
{
  _sec_size = get_sec_size<T>(Base<T>::_num_neurons);
  _num_secs = (Base<T>::_num_neurons) / _sec_size;
  _load_weight(weight_path);
}
*/

template <typename T>
Base<T>::Base(
  const std::fs::path& weight_path,
  const T bias,
  const size_t num_neurons,
  const size_t num_layers) : 
  _bias{bias},
  _num_neurons{num_neurons},
  _num_layers{num_layers} {

  std::cout << "Base constructor\n";
  /* 
  sycl::platform platform = sycl::platform::get_platforms()[0];
  sycl::device device = platform.get_devices()[0];
  sycl::queue queue  = sycl::queue(device);
   
  size_t shared_mem_size = device.get_info<sycl::info::device::local_mem_size>();
  queue.submit([](sycl::handler& handler) {
    handler.single_task([](){
    });  
  }); 
  */
  tf::Executor executor;
  tf::Taskflow taskflow;
   
  size_t shared_mem_size = queue.get_device().get_info<sycl::info::device::local_mem_size>();
  size_t nn = Base<T>::_num_neurons;
  size_t& ss = _sec_size;
  size_t& ns = _num_secs;
  /*
  queue.submit([](sycl::handler& handler){
    handler.single_task([](){});
  });
  */
  

  tf::Task sharedmemsize = taskflow.emplace_on([&](tf::syclFlow& sf){
    sf.single_task([](){}) ;
  }, queue).name("sharedmemsize");

  
  tf::Task secsize = taskflow.emplace([nn,shared_mem_size, &ss, &ns]() mutable {
    ss = get_sec_size<T>(nn, shared_mem_size);
    ns = nn / ss;
  }).name("secsize");
  /* 
  tf::Task numsecs = taskflow.emplace([nn, ss, &ns]() mutable {
    ns = get_num_sec<T>(nn, ss);
  }).name("numsecs");
  */
   
  tf::Task loadweight = taskflow.emplace([=](){
    _load_weight(weight_path); 
  }).name("loadweight");
  
  sharedmemsize.precede(secsize);
  //secsize.precede(numsecs);
  secsize.precede(loadweight);

  executor.run(taskflow).wait();  // run the  
  //taskflow.dump(std::cout); 
  
  /* 
  std::cout << "shared memory size = " << shared_mem_size << '\n'; 
  std::cout << "_sec_size = " << _sec_size << ", ss = " << ss << '\n';
  std::cout << "_num_secs = " << _num_secs << ", ns = " << ns << '\n';
  std::cout << "nn = " << nn << '\n';
  */
}




template <typename T>
Base<T>::~Base() {
  //sycl::free(_host_pinned_weight, queue);
  //cudaFreeHost(_host_pinned_weight);
  //checkCuda(cudaFreeHost(_host_pinned_weight));
}



template <typename T>
void Base<T>::_load_weight(const std::fs::path& weight_path) {
  log("Loading the weight......");
  tic();

  //std::cout << "_num_layers = " << _num_layers << '\n'; 
  //std::cout << "_num_neurons = " << _num_neurons << '\n';
  
  _max_nnz = find_max_nnz_binary(
               weight_path,
               _num_layers,
               _num_neurons
             );
  
  //////////// _max_nnz should not be hardcode
  _max_nnz = 32768;

  //std::cout << " _max_nnz = " << _max_nnz << "\n\n"; 
  // total length of row and col index
  // value index should consider sizeof(T)
  _p_w_index_len  = _num_neurons * _num_secs + _max_nnz + 1;

  //std::cout << "\n\n _p_w_index_len = " << _p_w_index_len << "\n\n"; 
  //handle aligned
  if((sizeof(int) * _p_w_index_len) % sizeof(T) != 0) {
    ++_pad;
  }

  _pp_w_index_len = _p_w_index_len + _pad;
  
  //std::cout << "\n\n _pad = " << _pad  << "\n\n"; 
  //std::cout << "\n\n _pp_w_index_len = " << _pp_w_index_len << "\n\n"; 

  //pad packed weight length
  //max_nnz should be even, otherwis it needs to be padded
  _pp_wlen = _pp_w_index_len + (sizeof(T) / sizeof(int)) * _max_nnz;

  //pad packed weight size
  _pp_wsize = sizeof(int) * (_pp_w_index_len) + sizeof(T) * _max_nnz;
  
  
  //checkCuda(cudaMallocHost(
  //  (void**)&_host_pinned_weight,
  //  _pp_wsize * _num_layers
  //));
  //std::cout << "_pp_wsize = " << _pp_wsize << '\n';
  
  /* 
  _host_pinned_weight = static_cast<int*>(sycl::malloc_host(
    _pp_wsize * _num_layers,
    queue
  ));
  */

  _host_pinned_weight = sycl::malloc_shared<int>(_pp_wsize*_num_layers/sizeof(int), queue); 

  queue.memset(_host_pinned_weight, 0, _pp_wsize * _num_layers).wait();

  //_host_pinned_weight = new int[_pp_wsize/sizeof(int) * _num_layers];
  
  //std::memset(
  //  _host_pinned_weight,
  //  0,
  //  _pp_wsize * _num_layers
  //);
 
   
  read_weight_binary<T>(
    weight_path,
    _num_neurons,
    _max_nnz,
    _num_layers,
    _num_secs,
    _pad,
    _host_pinned_weight
  );
  
  toc();
  log("Finish reading DNN layers with ", duration(), " ms", "\n");
}



template <typename T>
template <typename... ArgsT>
void Base<T>::log(ArgsT&&... args) const {
  _cout(std::forward<ArgsT>(args)...);
}



template<typename T>
void Base<T>::tic() {
  _tic = std::chrono::steady_clock::now();
  _enable_toc = true;
}

template<typename T>
void Base<T>::toc() {
  if(_enable_toc) {
    _toc = std::chrono::steady_clock::now();
    _enable_toc = false;
    _enable_counter = true;
    return;
  }
  throw std::runtime_error("Error counter. Checkout the order of counter function\n");
}



template<typename T>
auto Base<T>::duration() {
  if(_enable_counter) {
    _enable_counter = false;
    return std::chrono::duration_cast<std::chrono::milliseconds>(_toc - _tic).count();
  }
  throw std::runtime_error("Error ocunter. Checkout the order of counter functions\n");
}



template <typename T>
template <typename L>
void Base<T>::_cout(L&& last) const {
  std::cout << last << std::flush;
}

template <typename T>
template <typename First, typename... Remain>
void Base<T>::_cout(First&& item, Remain&&... remain) const {
  std::cout << item;
  _cout(std::forward<Remain>(remain)...);
}



/*
template <typename T>
size_t Base<T>::num_neurons() const {
   return _num_neurons; 
}

template <typename T>
size_t Base<T>::num_layers() const { 
  return _num_layers; 
}
*/




}  // end of namespace snig
