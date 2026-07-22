#pragma once
#include <nvtx3/nvToolsExt.h>
#include <string>
#include <iostream>

namespace ppu {
namespace fmha {

class FmhaProfParam {
public:

FmhaProfParam() {}

void initialize_args(const std::string& op_name, bool dir) {

  op_name_ = op_name;
  op_dir_ = dir ? "Forward" : "Backward";
  add_argument("data_type");     // string
  add_argument("batch_size");    // int32_t
  add_argument("num_heads");
  add_argument("num_heads_k");
  add_argument("head_dim");
  add_argument("head_dim_value");
  add_argument("seqlen_q");
  add_argument("seqlen_k");
  add_argument("custom_mask");
  add_argument("dropout");       //float
  add_argument("scale");
  add_argument("is_fixed_seqs"); //bool
}

template <typename T>
void add_mha_params(const std::string& key, const T& val) {
  if (args_.find(key) == args_.end()) {
    std::cout << "[" << key << "] don not exists." << std::endl;
    throw std::runtime_error("Add value fail.");
  }
  args_.at(key) = val_to_string(val);
}

void set_flashattn_params(const std::string& op_name, bool dir,
                          std::string data_type, int32_t batch_size,
                          int32_t num_heads, int32_t num_heads_k,
                          int32_t head_dim, int32_t head_dim_value,
                          int32_t seqlen_q, int32_t seqlen_k,
                          int32_t custom_mask, float dropout, float scale,
                          bool is_fixed_seqs){

  initialize_args(op_name, dir);
  add_mha_params("data_type", data_type);

  add_mha_params("batch_size", batch_size);
  add_mha_params("num_heads", num_heads);
  add_mha_params("num_heads_k", num_heads_k);
  add_mha_params("head_dim", head_dim);
  add_mha_params("head_dim_value", head_dim_value);
  add_mha_params("seqlen_q", seqlen_q);
  add_mha_params("seqlen_k", seqlen_k);

  add_mha_params("custom_mask", custom_mask);

  if (dropout > 0.0) {
    add_mha_params("dropout", dropout);
  }

  if (std::abs(scale * scale * head_dim - 1) > 1e-3) {
    // in general, scale can be get form inner computation.
    add_mha_params("scale", scale);
  }

  add_mha_params("is_fixed_seqs", is_fixed_seqs);
}

std::string format() {
  std::stringstream ss;

  ss << "[FMHA] --format=" << op_name_ << ',' << op_dir_;
  for(auto& iter: args_) {
    if (iter.second != ""){
      ss << ',' << iter.first << ':' << iter.second;
    }
  }
  ss << '.';
  return ss.str();
}

private:

std::string val_to_string(int32_t val) {
  return std::to_string(val);
}

std::string val_to_string(float val) {
  std::stringstream float_str;
  float_str << std::fixed << std::setprecision(4) << val;
  return float_str.str();
}

std::string val_to_string(bool val) {
  return std::to_string(int32_t(val));
}

std::string val_to_string(const std::string& val) {
  return val;
}

void add_argument(const std::string& name) {
  std::string init_val = "";
  if (args_.find(name) != args_.end()) {
    std::cout << "[" << name << "] already exists." << std::endl;
    throw std::runtime_error("Add argument fail.");
  }

  args_.insert(std::make_pair(name, init_val));
}

protected:
  std::string op_name_;
  std::string op_dir_;
  std::unordered_map<std::string, std::string> args_;
};


class ProfilingInterface {
public:
  ProfilingInterface(ProfilingInterface const&) = delete;
  void operator=(ProfilingInterface const&) = delete;

  static ProfilingInterface& Instance() {
    static ProfilingInterface instance;
    return instance;
  }

  bool get_op_info() {
    return show_params_ || use_nvtx_;
  }

  void instrument(bool start, FmhaProfParam &fmha_params) {
    if (!get_op_info()){
      return;
    }

    if (start) {
      std::string op_name = fmha_params.format();
      if (show_params_) {
        std::cout << op_name << std::endl;
      }
      if (use_nvtx_) {
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = op_name.c_str();
        nvtxDomainRangePushEx(domain_, &eventAttrib);
      }
    } else {
      if (use_nvtx_) {
        nvtxDomainRangePop(domain_);
      }
    } // if start

  }

private:
  ProfilingInterface() {
    // TODO: add print log
    domain_ = nvtxDomainCreateA("fmha");
    use_nvtx_ = false;
    show_params_ = false;

    char *pEnv_perf = std::getenv("PPU_LIB_PERF_INSTRUMENT");
    if (pEnv_perf && isdigit(*pEnv_perf)) {
      int value = std::stoi(std::string(pEnv_perf));
      if (value == 0) {
        use_nvtx_ = false;
      } else if (value == 1) {
        use_nvtx_ = true;
      } else {
        printf("Invalid value for PPU_LIB_PERF_INSTRUMENT : %d\n", value);
      }
    }

    char *pEnv_old = std::getenv("PPU_FLASH_ATTENTION_PRINT");
    if (pEnv_old && isdigit(*pEnv_old)) {
      int value = std::stoi(std::string(pEnv_old));
      if (value == 1) {
          printf("WARNING: PPU_FLASH_ATTENTION_PRINT is discarded. prease use PPU_LIB_SHOW_PARAMS! \n");
      }
    }
    char *pEnv_params = std::getenv("PPU_LIB_SHOW_PARAMS");
    if (pEnv_params && isdigit(*pEnv_params)) {
      int value = std::stoi(std::string(pEnv_params));
      if (value == 0) {
        show_params_ = false;
      } else if (value == 1) {
        show_params_ = true;
      } else {
        printf("Invalid value for PPU_LIB_SHOW_PARAMS : %d\n", value);
      }
    }

  }

  ~ProfilingInterface() {
    nvtxDomainDestroy(domain_);
  }

  bool use_nvtx_;
  bool show_params_;
  nvtxDomainHandle_t domain_;

};

} // namespace fmha
} // namespace ppu