/* Copyright 2023-2024 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mirage/utils/rocm_helper.h"

namespace mirage {
namespace utils {

rocblas_datatype to_cuda_datatype(mirage::type::DataType type) {
  switch (type) {
    case mirage::type::DT_FLOAT16:
      return rocblas_datatype_f16_r;
    case mirage::type::DT_FLOAT32:
      return rocblas_datatype_f32_r;
    case mirage::type::DT_DOUBLE:
      return rocblas_datatype_f64_r;
    default:
      assert(false && "Unsupported rocblas data type");
  }
  return rocblas_datatype_f16_r;
}

size_t get_max_shared_mem() {
  int device;
  hipGetDevice(&device);
  hipDeviceProp_t deviceProps;
  hipGetDeviceProperties(&deviceProps, device);
  return deviceProps.sharedMemPerBlock;
}

} // namespace utils
} // namespace mirage
