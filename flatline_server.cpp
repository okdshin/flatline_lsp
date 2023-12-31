#include <json/json.h>
#include <llama.h>
#include <ostream>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string_view>
#include <tuple>

#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>

#define CPPHTTPLIB_THREAD_POOL_COUNT 1
#include <httplib.h>
#undef CPPHTTPLIB_THREAD_POOL_COUNT

namespace {
std::shared_ptr<spdlog::logger> logger() {
  static auto logger_ = spdlog::stdout_color_mt("flatline");
  return logger_;
}
} // namespace

namespace {
struct llama_model_deleter {
  void operator()(llama_model *model) noexcept { llama_free_model(model); }
};
struct llama_context_deleter {
  void operator()(llama_context *context) noexcept { llama_free(context); }
};
using unique_llama_model = std::unique_ptr<llama_model, llama_model_deleter>;
using unique_llama_context =
    std::unique_ptr<llama_context, llama_context_deleter>;

class llama_cpp_model {
public:
  static llama_cpp_model load_from_file(std::string const &model_file_path,
                                        size_t n_threads, size_t n_gpu_layers) {

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    unique_llama_model model(
        llama_load_model_from_file(model_file_path.c_str(), model_params));
    if (!model) {
      throw std::runtime_error("wrong model_path");
    }

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.seed = 1234;
    ctx_params.n_ctx = 2048; // TODO
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    unique_llama_context ctx(
        llama_new_context_with_model(model.get(), ctx_params));
    if (!ctx) {
      throw std::runtime_error("failed to create context with model");
    }

    return llama_cpp_model(std::move(model), std::move(ctx));
  }

  std::vector<float> calc_next_token_logits(std::vector<int> const &input_ids) {
    llama_batch batch = llama_batch_init(2048, 0); // TODO
    if (is_first(input_ids)) {
      logger()->info("no kv_cache");
      llama_kv_cache_tokens_rm(ctx_.get(), -1, -1);
      batch.n_tokens = input_ids.size();
      for (size_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = input_ids[i];
        batch.pos[i] = i;
        batch.seq_id[i] = 0;
        batch.logits[i] = false;
      }
      batch.logits[batch.n_tokens - 1] = true;
    } else {
      logger()->info("using kv_cache");
      batch.token[0] = input_ids[input_ids.size() - 1];
      batch.pos[0] = input_ids.size() - 1;
      batch.seq_id[0] = 0;
      batch.logits[0] = true;
      batch.n_tokens = 1;
    }
    // if (auto result = llama_decode(ctx_, batch); result != 0) {
    if (auto result = llama_decode(ctx_.get(), batch); result < 0) {
      throw std::runtime_error("llama_decode failed " + std::to_string(result));
    }
    auto *logits_data = llama_get_logits_ith(ctx_.get(), batch.n_tokens - 1);
    std::vector<float> logits(vocab_size_);
    std::copy(logits_data, logits_data + vocab_size_, logits.begin());
    return logits;
  }

private:
  llama_cpp_model(unique_llama_model &&model, unique_llama_context &&ctx)
      : model_(std::move(model)), ctx_(std::move(ctx)) {}

  bool is_first(std::vector<int> const &input_ids) {
    static std::vector<int> input_ids_before_backup = std::vector<int>();
    std::vector<int> input_ids_before = input_ids_before_backup;
    input_ids_before_backup = input_ids;
    if (input_ids_before.size() > input_ids.size()) {
      return true;
    }
    for (size_t i = 0; i < input_ids_before.size(); ++i) {
      if (input_ids_before[i] != input_ids[i]) {
        return true;
      }
    }
    return false;
  }

  size_t vocab_size_ = 51200; // TODO load from model data
  unique_llama_model model_;
  unique_llama_context ctx_;
};
} // namespace

std::optional<Json::Value> try_to_parse_json(httplib::Request const &req) {
  Json::CharReaderBuilder builder;
  const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  Json::Value root;
  JSONCPP_STRING err;
  std::string_view body = req.body;
  logger()->info("request {}", body);
  if (!reader->parse(body.data(), body.data() + body.size(), &root, &err)) {
    return std::nullopt;
  }
  return root;
}

std::vector<int> get_request_data(Json::Value const &root) {
  Json::Value input_tokens_value = root["input_tokens"];
  std::vector<int> input_tokens(input_tokens_value.size());
  std::transform(input_tokens_value.begin(), input_tokens_value.end(),
                 input_tokens.begin(),
                 [](Json::Value const &e) { return e.asInt(); });
  return input_tokens;
}

std::string make_response_json(std::vector<float> const &next_token_logits) {
  Json::Value response;
  Json::Value next_token_logits_value = Json::Value(Json::arrayValue);
  for (float logit : next_token_logits) {
    next_token_logits_value.append(logit);
  }
  response["next_token_logits"] = next_token_logits_value;
  Json::FastWriter json_fast_writer;
  return json_fast_writer.write(response);
}

#include <structopt/app.hpp>
struct app_options {
  std::optional<std::string> host = "localhost";
  std::optional<int> port = 57045;
  std::optional<std::string> model_path;
  std::optional<bool> numa = true;
  std::optional<int> n_threads = -1;
  std::optional<int> n_gpu_layers = 35;
};
STRUCTOPT(app_options, port, model_path, numa, n_threads, n_gpu_layers);

int main(int argc, char **argv) {
  auto options = structopt::app("flatline").parse<app_options>(argc, argv);
  if (!options.model_path) {
    throw std::runtime_error("wrong model_path");
  }
  const size_t server_thread_num = 1; // Must be 1
  const size_t max_thread_num = std::thread::hardware_concurrency();
  const size_t infer_thread_num =
      *options.n_threads < 0 ? max_thread_num / 2 : *options.n_threads;

  llama_backend_init(*options.numa);

  auto model = llama_cpp_model::load_from_file(
      *options.model_path, infer_thread_num, *options.n_gpu_layers);
  logger()->info("model loading finished");

  httplib::Server server;
  server.Get("/", [](httplib::Request const &req, httplib::Response &res) {
    res.set_content("Flatline backend server is available", "text/plain");
  });
  server.Get("/config", [&options](httplib::Request const &req,
                                   httplib::Response &res) {
    Json::Value config;
    config["port"] = *options.port;
    config["model_path"] = *options.model_path;
    config["numa"] = *options.numa;
    config["n_threads"] = *options.n_threads;
    config["n_gpu_layers"] = *options.n_gpu_layers;
    Json::FastWriter json_fast_writer;
    res.set_content(json_fast_writer.write(config), "application/json");
  });
  constexpr int status_bad_request = 400;
  std::mutex computing_resource_mutex;
  server.Post("/v1/calc_next_token_logits", [&model, &computing_resource_mutex](
                                                httplib::Request const &req,
                                                httplib::Response &res) {
    // Header check
    if (req.get_header_value("Content-type") != "application/json") {
      res.status = status_bad_request;
      res.set_content("\"Content-type\" must be \"application/json\"",
                      "text/plain");
      logger()->info("Content-type is not application/json");
      return;
    }

    // Data check & parse
    std::optional<Json::Value> root_opt = try_to_parse_json(req);
    if (!root_opt) {
      res.status = status_bad_request;
      res.set_content("JSON data is broken", "text/plain");
      logger()->info("JSON data is broken");
      return;
    }
    Json::Value const &root = *root_opt;
    std::vector<int> input_tokens = get_request_data(root);

    std::vector<float> next_token_logits;
    {
      // lock
      std::unique_lock lock(computing_resource_mutex);

      // Calc next token logits
      next_token_logits = model.calc_next_token_logits(input_tokens);
    }

    // Send response
    std::string response_json = make_response_json(next_token_logits);
    res.set_content(response_json.c_str(), "application/json");
    logger()->info("sent response {}",
                   std::string(response_json.c_str()).substr(0, 128) + "...");
  });
  server.listen(*options.host, *options.port);

  llama_backend_free();

  return 0;
}
