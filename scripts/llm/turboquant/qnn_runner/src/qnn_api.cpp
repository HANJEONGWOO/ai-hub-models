// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#include "qnn_api.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>

#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

namespace tqrun {
namespace {

[[noreturn]] void fail(const std::string& what, Qnn_ErrorHandle_t err = QNN_SUCCESS) {
  throw std::runtime_error(what + (err == QNN_SUCCESS ? "" : " (QNN error " + std::to_string(err) + ")"));
}

void logCallback(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args) {
  if (level > QNN_LOG_LEVEL_WARN) return;
  std::fprintf(stderr, "[qnn %d] ", static_cast<int>(level));
  std::vfprintf(stderr, fmt, args);
  std::fprintf(stderr, "\n");
}

const Qnn_TensorV1_t& v1(const Qnn_Tensor_t& t) { return t.v1; }  // V2 shares the V1 prefix.

TensorMeta copyTensor(const Qnn_Tensor_t& src) {
  if (src.version != QNN_TENSOR_VERSION_1 && src.version != QNN_TENSOR_VERSION_2) {
    fail("unsupported tensor version " + std::to_string(src.version));
  }
  const Qnn_TensorV1_t& s = v1(src);
  TensorMeta meta;
  meta.name = s.name ? s.name : "";
  meta.dims.assign(s.dimensions, s.dimensions + s.rank);
  meta.dataType = s.dataType;
  const Qnn_QuantizeParams_t& q = s.quantizeParams;
  if (q.encodingDefinition == QNN_DEFINITION_DEFINED) {
    if (q.quantizationEncoding != QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
      fail("tensor " + meta.name + " uses a non per-tensor quantization encoding");
    }
    meta.quantized = true;
    meta.scale = q.scaleOffsetEncoding.scale;
    meta.offset = q.scaleOffsetEncoding.offset;
  }
  Qnn_Tensor_t proto = QNN_TENSOR_INIT;
  proto.version = QNN_TENSOR_VERSION_1;
  proto.v1.id = s.id;
  proto.v1.type = s.type;
  proto.v1.dataFormat = s.dataFormat;
  proto.v1.dataType = s.dataType;
  proto.v1.quantizeParams = s.quantizeParams;
  proto.v1.rank = s.rank;
  proto.v1.memType = QNN_TENSORMEMTYPE_RAW;
  meta.prototype = proto;
  return meta;
}

}  // namespace

size_t TensorMeta::elementBytes() const {
  switch (dataType) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
      return 1;
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_FLOAT_16:
      return 2;
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
    case QNN_DATATYPE_FLOAT_32:
      return 4;
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
    case QNN_DATATYPE_FLOAT_64:
      return 8;
    default:
      fail("tensor " + name + " has unsupported data type " + std::to_string(dataType));
  }
}

size_t TensorMeta::numElements() const {
  size_t n = 1;
  for (uint32_t d : dims) n *= d;
  return n;
}

QnnRuntime::QnnRuntime(const std::string& backendLib, const std::string& systemLib) {
  backendLib_ = dlopen(backendLib.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!backendLib_) fail(std::string("dlopen backend: ") + dlerror());
  systemLib_ = dlopen(systemLib.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!systemLib_) fail(std::string("dlopen system: ") + dlerror());

  using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
  using GetSysProviders = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);
  auto getProviders = reinterpret_cast<GetProviders>(dlsym(backendLib_, "QnnInterface_getProviders"));
  auto getSysProviders = reinterpret_cast<GetSysProviders>(dlsym(systemLib_, "QnnSystemInterface_getProviders"));
  if (!getProviders || !getSysProviders) fail("QNN provider symbols not found");

  const QnnInterface_t** providers = nullptr;
  uint32_t count = 0;
  if (getProviders(&providers, &count) != QNN_SUCCESS || count == 0) fail("no QNN backend providers");
  for (uint32_t i = 0; i < count; ++i) {
    const auto& v = providers[i]->apiVersion.coreApiVersion;
    if (v.major == QNN_API_VERSION_MAJOR && v.minor >= QNN_API_VERSION_MINOR) {
      api_ = &providers[i]->QNN_INTERFACE_VER_NAME;
      const auto& b = providers[i]->apiVersion.backendApiVersion;
      backendVersion_ = std::to_string(v.major) + "." + std::to_string(v.minor) + "." + std::to_string(v.patch) +
                        "/htp " + std::to_string(b.major) + "." + std::to_string(b.minor) + "." +
                        std::to_string(b.patch);
      break;
    }
  }
  if (!api_) fail("no QNN backend provider compatible with the headers");

  const QnnSystemInterface_t** sysProviders = nullptr;
  count = 0;
  if (getSysProviders(&sysProviders, &count) != QNN_SUCCESS || count == 0) fail("no QNN system providers");
  for (uint32_t i = 0; i < count; ++i) {
    const auto& v = sysProviders[i]->systemApiVersion;
    if (v.major == QNN_SYSTEM_API_VERSION_MAJOR && v.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
      sysApi_ = &sysProviders[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
      break;
    }
  }
  if (!sysApi_) fail("no QNN system provider compatible with the headers");

  Qnn_ErrorHandle_t err = api_->logCreate(logCallback, QNN_LOG_LEVEL_WARN, &log_);
  if (err != QNN_SUCCESS) fail("logCreate", err);
  err = api_->backendCreate(log_, nullptr, &backend_);
  if (err != QNN_SUCCESS) fail("backendCreate", err);
  if (api_->deviceCreate) {
    err = api_->deviceCreate(log_, nullptr, &device_);
    if (err != QNN_SUCCESS) fail("deviceCreate", err);
  }
}

QnnRuntime::~QnnRuntime() {
  if (hasPowerConfig_ && api_ && api_->deviceGetInfrastructure) {
    QnnDevice_Infrastructure_t infra = nullptr;
    if (api_->deviceGetInfrastructure(&infra) == QNN_SUCCESS && infra) {
      static_cast<QnnHtpDevice_Infrastructure_t*>(infra)->perfInfra.destroyPowerConfigId(powerConfigId_);
    }
  }
  if (profile_) api_->profileFree(profile_);
  if (device_) api_->deviceFree(device_);
  if (backend_) api_->backendFree(backend_);
  if (log_) api_->logFree(log_);
  if (systemLib_) dlclose(systemLib_);
  if (backendLib_) dlclose(backendLib_);
}

bool QnnRuntime::setBurstPower() {
  if (!api_->deviceGetInfrastructure) return false;
  QnnDevice_Infrastructure_t infra = nullptr;
  if (api_->deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) return false;
  auto* htp = static_cast<QnnHtpDevice_Infrastructure_t*>(infra);
  if (htp->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return false;
  QnnHtpDevice_PerfInfrastructure_t& perf = htp->perfInfra;
  if (perf.createPowerConfigId(0, 0, &powerConfigId_) != QNN_SUCCESS) return false;
  hasPowerConfig_ = true;

  QnnHtpPerfInfrastructure_PowerConfig_t dcvs{};
  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  auto& d = dcvs.dcvsV3Config;
  d.contextId = powerConfigId_;
  d.setDcvsEnable = 1;
  d.dcvsEnable = 0;
  d.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  d.setSleepLatency = 1;
  d.sleepLatency = 40;
  d.setSleepDisable = 1;
  d.sleepDisable = 1;
  d.setBusParams = 1;
  d.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  d.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  d.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  d.setCoreParams = 1;
  d.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  d.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  d.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

  QnnHtpPerfInfrastructure_PowerConfig_t latency{};
  latency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  latency.rpcControlLatencyConfig = 100;
  QnnHtpPerfInfrastructure_PowerConfig_t polling{};
  polling.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  polling.rpcPollingTimeConfig = 9999;

  const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = {&dcvs, &latency, &polling, nullptr};
  return perf.setPowerConfig(powerConfigId_, configs) == QNN_SUCCESS;
}

void QnnRuntime::execute(GraphInfo& graph, std::vector<Qnn_Tensor_t>& inputs, std::vector<Qnn_Tensor_t>& outputs) {
  // A context loaded with a profile handle must be executed with it (else QNN_GRAPH_ERROR_SET_PROFILE).
  Qnn_ErrorHandle_t err = api_->graphExecute(graph.handle, inputs.data(), static_cast<uint32_t>(inputs.size()),
                                             outputs.data(), static_cast<uint32_t>(outputs.size()), profile_, nullptr);
  if (err != QNN_SUCCESS) fail("graphExecute " + graph.name, err);
}

void QnnRuntime::enableDetailedProfiling() {
  if (profile_) return;
  Qnn_ErrorHandle_t err = api_->profileCreate(backend_, QNN_PROFILE_LEVEL_DETAILED, &profile_);
  if (err != QNN_SUCCESS) fail("profileCreate", err);
}

std::vector<ProfileEvent> QnnRuntime::executeProfiled(GraphInfo& graph, std::vector<Qnn_Tensor_t>& inputs,
                                                      std::vector<Qnn_Tensor_t>& outputs) {
  if (!profile_) fail("executeProfiled requires enableDetailedProfiling before loading contexts");
  Qnn_ErrorHandle_t err = api_->graphExecute(graph.handle, inputs.data(), static_cast<uint32_t>(inputs.size()),
                                             outputs.data(), static_cast<uint32_t>(outputs.size()), profile_, nullptr);
  if (err != QNN_SUCCESS) fail("graphExecute (profiled) " + graph.name, err);
  std::vector<ProfileEvent> events;
  std::function<void(const QnnProfile_EventId_t*, uint32_t, int)> walk =
      [&](const QnnProfile_EventId_t* ids, uint32_t count, int depth) {
        for (uint32_t i = 0; i < count; ++i) {
          QnnProfile_EventData_t data{};
          if (api_->profileGetEventData(ids[i], &data) != QNN_SUCCESS) continue;
          events.push_back({depth, data.type, data.unit, data.value, data.identifier ? data.identifier : ""});
          const QnnProfile_EventId_t* sub = nullptr;
          uint32_t numSub = 0;
          if (api_->profileGetSubEvents(ids[i], &sub, &numSub) == QNN_SUCCESS && numSub) walk(sub, numSub, depth + 1);
        }
      };
  const QnnProfile_EventId_t* ids = nullptr;
  uint32_t count = 0;
  // The handle holds only the most recent execute's events.
  if (api_->profileGetEvents(profile_, &ids, &count) == QNN_SUCCESS) walk(ids, count, 0);
  return events;
}

std::string QnnRuntime::backendVersion() const { return backendVersion_; }

ContextBinary::ContextBinary(QnnRuntime& rt, const std::string& path, const std::vector<std::string>& graphNames)
    : rt_(rt), path_(path) {
  auto start = std::chrono::steady_clock::now();
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) fail("open " + path);
  struct stat st {};
  fstat(fd, &st);
  fileBytes_ = static_cast<uint64_t>(st.st_size);
  void* data = mmap(nullptr, fileBytes_, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (data == MAP_FAILED) fail("mmap " + path);

  const auto& sys = rt_.systemApi();
  QnnSystemContext_Handle_t sysCtx = nullptr;
  if (sys.systemContextCreate(&sysCtx) != QNN_SUCCESS) fail("systemContextCreate");
  const QnnSystemContext_BinaryInfo_t* info = nullptr;
  Qnn_ErrorHandle_t err = sys.systemContextGetMetaData(sysCtx, data, fileBytes_, &info);
  if (err != QNN_SUCCESS || !info) fail("systemContextGetMetaData " + path, err);

  uint32_t numGraphs = 0;
  QnnSystemContext_GraphInfo_t* graphs = nullptr;
  switch (info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
      numGraphs = info->contextBinaryInfoV1.numGraphs;
      graphs = info->contextBinaryInfoV1.graphs;
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
      numGraphs = info->contextBinaryInfoV2.numGraphs;
      graphs = info->contextBinaryInfoV2.graphs;
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
      numGraphs = info->contextBinaryInfoV3.numGraphs;
      graphs = info->contextBinaryInfoV3.graphs;
      break;
    default:
      fail("unsupported context binary info version");
  }
  for (uint32_t g = 0; g < numGraphs; ++g) {
    // Graph info V1/V2/V3 share the leading name/input/output fields.
    const QnnSystemContext_GraphInfoV1_t& gi = graphs[g].graphInfoV1;
    std::string name = gi.graphName ? gi.graphName : "";
    bool wanted = false;
    for (const auto& w : graphNames) wanted |= (w == name);
    if (!wanted) continue;
    auto info = std::make_unique<GraphInfo>();
    info->name = name;
    for (uint32_t i = 0; i < gi.numGraphInputs; ++i) info->inputs.push_back(copyTensor(gi.graphInputs[i]));
    for (uint32_t i = 0; i < gi.numGraphOutputs; ++i) info->outputs.push_back(copyTensor(gi.graphOutputs[i]));
    graphs_[name] = std::move(info);
  }
  sys.systemContextFree(sysCtx);
  for (const auto& w : graphNames) {
    if (!graphs_.count(w)) fail("graph " + w + " not found in " + path);
  }

  err = rt_.api().contextCreateFromBinary(rt_.backend(), rt_.device(), nullptr, data, fileBytes_, &context_, rt_.profile());
  munmap(data, fileBytes_);
  if (err != QNN_SUCCESS) fail("contextCreateFromBinary " + path, err);
  for (auto& [name, graph] : graphs_) {
    err = rt_.api().graphRetrieve(context_, name.c_str(), &graph->handle);
    if (err != QNN_SUCCESS) fail("graphRetrieve " + name, err);
  }
  loadSeconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

ContextBinary::~ContextBinary() {
  if (context_) rt_.api().contextFree(context_, nullptr);
}

GraphInfo& ContextBinary::graph(const std::string& name) {
  auto it = graphs_.find(name);
  if (it == graphs_.end()) fail("graph " + name + " not loaded from " + path_);
  return *it->second;
}

}  // namespace tqrun
