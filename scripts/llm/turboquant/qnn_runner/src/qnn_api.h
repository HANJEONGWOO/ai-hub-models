// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
// Thin wrapper over the public QNN C API: backend/device lifetime, context
// binaries with their graph I/O metadata, graph execution and HTP power votes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

namespace tqrun {

struct TensorMeta {
  Qnn_Tensor_t prototype{};  // copied from metadata; name/dims point into this struct
  std::string name;
  std::vector<uint32_t> dims;
  Qnn_DataType_t dataType = QNN_DATATYPE_UNDEFINED;
  bool quantized = false;
  double scale = 1.0;
  int32_t offset = 0;

  size_t elementBytes() const;
  size_t numElements() const;
  size_t bytes() const { return elementBytes() * numElements(); }
};

struct ProfileEvent {
  int depth = 0;
  uint32_t type = 0;
  uint32_t unit = 0;
  uint64_t value = 0;
  std::string identifier;
};

struct GraphInfo {
  std::string name;
  Qnn_GraphHandle_t handle = nullptr;
  std::vector<TensorMeta> inputs;
  std::vector<TensorMeta> outputs;
};

class QnnRuntime;

class ContextBinary {
 public:
  ContextBinary(QnnRuntime& rt, const std::string& path, const std::vector<std::string>& graphNames);
  ~ContextBinary();
  ContextBinary(const ContextBinary&) = delete;
  ContextBinary& operator=(const ContextBinary&) = delete;

  GraphInfo& graph(const std::string& name);
  const std::string& path() const { return path_; }
  double loadSeconds() const { return loadSeconds_; }
  uint64_t fileBytes() const { return fileBytes_; }

 private:
  QnnRuntime& rt_;
  std::string path_;
  Qnn_ContextHandle_t context_ = nullptr;
  std::map<std::string, std::unique_ptr<GraphInfo>> graphs_;
  double loadSeconds_ = 0.0;
  uint64_t fileBytes_ = 0;
};

class QnnRuntime {
 public:
  QnnRuntime(const std::string& backendLib, const std::string& systemLib);
  ~QnnRuntime();
  QnnRuntime(const QnnRuntime&) = delete;
  QnnRuntime& operator=(const QnnRuntime&) = delete;

  // Votes the HTP into performance mode with sleep disabled; returns false if unsupported.
  bool setBurstPower();
  // Register execution kernels before loading any dependent context binary.
  void registerOpPackage(const std::string& path, const std::string& provider);
  // Must be called before any context is created: HTP binds detailed profiling at load time.
  void enableDetailedProfiling();
  Qnn_ProfileHandle_t profile() const { return profile_; }
  void execute(GraphInfo& graph, std::vector<Qnn_Tensor_t>& inputs, std::vector<Qnn_Tensor_t>& outputs);
  // Executes with QNN detailed profiling and returns the flattened event tree.
  std::vector<ProfileEvent> executeProfiled(GraphInfo& graph, std::vector<Qnn_Tensor_t>& inputs,
                                            std::vector<Qnn_Tensor_t>& outputs);
  std::string backendVersion() const;

  const QNN_INTERFACE_VER_TYPE& api() const { return *api_; }
  const QNN_SYSTEM_INTERFACE_VER_TYPE& systemApi() const { return *sysApi_; }
  Qnn_BackendHandle_t backend() const { return backend_; }
  Qnn_DeviceHandle_t device() const { return device_; }

 private:
  void* backendLib_ = nullptr;
  void* systemLib_ = nullptr;
  const QNN_INTERFACE_VER_TYPE* api_ = nullptr;
  const QNN_SYSTEM_INTERFACE_VER_TYPE* sysApi_ = nullptr;
  Qnn_LogHandle_t log_ = nullptr;
  Qnn_BackendHandle_t backend_ = nullptr;
  Qnn_DeviceHandle_t device_ = nullptr;
  Qnn_ProfileHandle_t profile_ = nullptr;
  uint32_t powerConfigId_ = 0;
  bool hasPowerConfig_ = false;
  std::string backendVersion_;
};

}  // namespace tqrun
