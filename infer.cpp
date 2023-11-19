#include <iostream>
#include <cassert>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <llama.h>

namespace {

namespace py = pybind11;

class llama_cpp_model {
public:
  static std::unique_ptr<llama_cpp_model>
  load_from_file(std::string const &model_file_path, size_t n_threads) {

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 20;
    llama_model *model =
        llama_load_model_from_file(model_file_path.c_str(), model_params);

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.seed = 1234;
    ctx_params.n_ctx = 2048; // TODO
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch =
        n_threads; // params.n_threads_batch == -1 ? params.n_threads :
                   // params.n_threads_batch;
    llama_context *ctx = llama_new_context_with_model(model, ctx_params);

    return std::make_unique<llama_cpp_model>(
        llama_cpp_model(std::move(model), ctx));
  }

  py::array_t<float> calc_next_token_logits(py::array_t<int> const &input_ids,
                                            size_t vocab_size) {
    assert(input_ids.shape(0) == 1); // batch_size must be 1
    llama_batch batch = llama_batch_init(2048, 0); // TODO
    if (is_first(input_ids)) {
      //py::print("FIRST, input_ids = ", input_ids);
      llama_kv_cache_tokens_rm(ctx_, -1, -1);
      batch.n_tokens = input_ids.shape(1);
      for (size_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = *input_ids.data(0, i);
        batch.pos[i] = i;
        batch.seq_id[i] = 0;
        batch.logits[i] = false;
      }
      batch.logits[batch.n_tokens - 1] = true;
    } else {
      //py::print("input_ids = ", input_ids);
      batch.token[0] = *input_ids.data(0, input_ids.shape(1) - 1);
      batch.pos[0] = input_ids.shape(1) - 1;
      batch.seq_id[0] = 0;
      batch.logits[0] = true;
      batch.n_tokens = 1;
    }
    // if (auto result = llama_decode(ctx_, batch); result != 0) {
    if (auto result = llama_decode(ctx_, batch); result < 0) {
      throw std::runtime_error("llama_decode failed " + std::to_string(result));
    }
    auto *logits_data = llama_get_logits_ith(ctx_, batch.n_tokens - 1);
    py::array_t<float> logits(
        std::vector<size_t>{static_cast<size_t>(input_ids.shape(0)), 1u,
                            vocab_size},
        logits_data);
    //py::print("logits = ", logits);
    return logits;
  }

private:
  llama_cpp_model(llama_model *model, llama_context *ctx)
      : model_(model), ctx_(ctx) {}

  bool is_first(py::array_t<int> const &input_ids) {
      static py::array_t<int> input_ids_before_backup = py::array_t<int>();
      py::array_t<int> input_ids_before = input_ids_before_backup;
      input_ids_before_backup = input_ids;
      if(input_ids_before.ndim() != input_ids.ndim()) {
          return true;
      }
      if(input_ids_before.shape(0) != input_ids.shape(0)) {
          return true;
      }
      if(input_ids_before.shape(1) > input_ids.shape(1)) {
          return true;
      }
      for(size_t i = 0; i < input_ids_before.shape(0); ++i) {
          for(size_t j = 0; j < input_ids_before.shape(1); ++j) {
              if(*input_ids_before.data(i, j) != *input_ids.data(i, j)) {
                  return true;
              }
          }
      }
      return false;
  }

  llama_model *model_;
  llama_context *ctx_;
};

} // namespace

PYBIND11_MODULE(infer, m) {
  m.doc() = "infer module";

  m.def("load_model_from_file", &llama_cpp_model::load_from_file, "",
        py::arg("model_file_path"), py::arg("n_threads"));

  py::class_<llama_cpp_model, std::unique_ptr<llama_cpp_model>>(
      m, "llama_cpp_model")
      //.def(py::init<>()) // use load_model_from_file
      .def("calc_next_token_logits", &llama_cpp_model::calc_next_token_logits,
           py::arg("input_ids"), py::arg("vocab_size"));
}
