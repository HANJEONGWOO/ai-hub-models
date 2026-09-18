// SPDX-License-Identifier: BSD-3-Clause
// TurboQuant format-2, MSB-first 4-bit decoder. No rotation or norm reduction.
#include "HTP/QnnHtpCommon.h"
#include "HTP/core/qhpi.h"
#include "QnnOpPackage.h"
#include "QnnSdkBuildId.h"
#include <stdint.h>
#include <string.h>
#ifdef __hexagon__
#include "hvx_decode.h"
#endif

namespace {
constexpr const char* kPackage = "TurboQuantNative";
const char* kOps[] = {"Decode4"};
Qnn_ApiVersion_t apiVersion = QNN_HTP_API_VERSION_INIT;
Qnn_Version_t opVersion = {1, 0, 0};
QnnOpPackage_Info_t packageInfo = {kPackage, kOps, nullptr, 1, nullptr, 0,
    QNN_SDK_BUILD_ID, &apiVersion, nullptr, &opVersion, {0}};
bool initialized = false;

Qnn_ErrorHandle_t init(QnnOpPackage_GlobalInfrastructure_t) {
  if (initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
  initialized = true;
  return QNN_SUCCESS;
}
Qnn_ErrorHandle_t terminate() {
  initialized = false;
  return QNN_SUCCESS;
}
Qnn_ErrorHandle_t info(const QnnOpPackage_Info_t** value) {
  if (!initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
  if (!value) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
  *value = &packageInfo;
  return QNN_SUCCESS;
}
Qnn_ErrorHandle_t validate(Qnn_OpConfig_t op) {
  if (op.version != QNN_OPCONFIG_VERSION_1 || !op.v1.packageName ||
      !op.v1.typeName || strcmp(op.v1.packageName, kPackage) ||
      strcmp(op.v1.typeName, kOps[0]) || op.v1.numOfParams != 0 ||
      op.v1.numOfInputs != 3 || op.v1.numOfOutputs != 1)
    return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
  return QNN_SUCCESS;
}
Qnn_ErrorHandle_t create(QnnOpPackage_GraphInfrastructure_t,
                        QnnOpPackage_Node_t, QnnOpPackage_OpImpl_t*) {
  return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}
Qnn_ErrorHandle_t release(QnnOpPackage_OpImpl_t) {
  return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}
Qnn_ErrorHandle_t logLevel(QnnLog_Level_t level) {
  return level >= QNN_LOG_LEVEL_ERROR ? QNN_SUCCESS : QNN_LOG_ERROR_INVALID_ARGUMENT;
}
Qnn_ErrorHandle_t logInit(QnnLog_Callback_t callback, QnnLog_Level_t level) {
  return callback ? logLevel(level) : QNN_LOG_ERROR_INVALID_ARGUMENT;
}
Qnn_ErrorHandle_t logEnd() { return QNN_SUCCESS; }

uint32_t elements(const QHPI_Shape& shape) {
  uint32_t count = 1;
  for (uint32_t i = 0; i < shape.rank; ++i) count *= shape.dims[i];
  return count;
}

uint32_t decode(QHPI_RuntimeHandle* handle, uint32_t nout, QHPI_Tensor** outputs,
                uint32_t nin, const QHPI_Tensor* const* inputs) {
  if (nin != 3 || nout != 1) return QHPI_ERROR_FATAL;
  const auto packedShape = qhpi_tensor_shape(inputs[0]);
  const auto scaleShape = qhpi_tensor_shape(inputs[1]);
  const auto outShape = qhpi_tensor_shape(outputs[0]);
  if (packedShape.rank != 4 || scaleShape.rank != 4 || outShape.rank != 4 ||
      packedShape.dims[3] != 64 || scaleShape.dims[3] != 1 ||
      outShape.dims[3] != 128 || elements(qhpi_tensor_shape(inputs[2])) != 16 ||
      qhpi_element_type_size(qhpi_tensor_type(inputs[0])) != 1)
    return QHPI_ERROR_FATAL;
  for (unsigned i = 0; i < 3; ++i)
    if (packedShape.dims[i] != scaleShape.dims[i] ||
        packedShape.dims[i] != outShape.dims[i]) return QHPI_ERROR_FATAL;
  const auto* packed = static_cast<const uint8_t*>(qhpi_tensor_raw_data(inputs[0]));
  const auto* scale = static_cast<const __fp16*>(qhpi_tensor_raw_data(inputs[1]));
  const auto* table = static_cast<const __fp16*>(qhpi_tensor_raw_data(inputs[2]));
  auto* output = static_cast<__fp16*>(qhpi_tensor_raw_data(outputs[0]));
  if (!packed || !scale || !table || !output) return QHPI_ERROR_FATAL;
  const uint32_t rows = elements(scaleShape);
  const uint32_t slices = qhpi_num_slices(handle);
  const uint32_t slice = qhpi_slice_number(handle);
  const uint32_t begin = uint64_t(rows) * slice / slices;
  const uint32_t end = uint64_t(rows) * (slice + 1) / slices;
#ifdef __hexagon__
  tq_decode_hvx(packed + begin * 64,
      reinterpret_cast<const uint16_t*>(scale) + begin,
      reinterpret_cast<const uint16_t*>(table),
      reinterpret_cast<uint16_t*>(output) + begin * 128, end - begin);
#else
  for (uint32_t row = begin; row < end; ++row) {
    for (unsigned j = 0; j < 64; ++j) {
      const unsigned byte = packed[row * 64 + j];
      output[row * 128 + 2 * j] = float(table[byte >> 4]) * float(scale[row]);
      output[row * 128 + 2 * j + 1] = float(table[byte & 15]) * float(scale[row]);
    }
  }
#endif
  return QHPI_SUCCESS;
}

float cost(uint32_t, const QHPI_Tensor* const* inputs) {
  return 100.0f + 20.0f * elements(qhpi_tensor_shape(inputs[1]));
}
QHPI_Tensor_Signature_v1 inputSignatures[] = {
    {QHPI_ELEMENT_TYPE_ANY, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM},
    {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM},
    {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM}};
QHPI_Tensor_Signature_v1 outputSignatures[] = {
    {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM}};
QHPI_Kernel_v1 kernel = {};
QHPI_OpInfo_v1 op = {};
}  // namespace

extern "C" Qnn_ErrorHandle_t TurboQuantInterfaceProvider(QnnOpPackage_Interface_t* out) {
  if (!out) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
  *out = {};
  out->interfaceVersion = {1, 4, 0};
  out->v1_4.init = init;
  out->v1_4.terminate = terminate;
  out->v1_4.getInfo = info;
  out->v1_4.validateOpConfig = validate;
  out->v1_4.createOpImpl = create;
  out->v1_4.freeOpImpl = release;
  out->v1_4.logInitialize = logInit;
  out->v1_4.logSetLevel = logLevel;
  out->v1_4.logTerminate = logEnd;
  return QNN_SUCCESS;
}

extern "C" const char* qhpi_init() {
  kernel.function_name = "tq_decode4";
  kernel.function = decode;
  kernel.resources = QHPI_RESOURCE_HVX;
  kernel.multithreaded = true;
  kernel.min_inputs = 3;
  kernel.input_signature = inputSignatures;
  kernel.min_outputs = 1;
  kernel.output_signature = outputSignatures;
  kernel.cost_function = cost;
  op.name = "TurboQuantNative::Decode4";
  op.num_kernels = 1;
  op.kernels = &kernel;
  qhpi_register_ops_v1(1, &op, kPackage);
  return kPackage;
}
