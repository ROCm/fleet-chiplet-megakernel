#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  int n = 0;
  hipGetDeviceCount(&n);
  for (int i = 0; i < n; ++i) {
    char bus[64] = {0};
    hipDeviceGetPCIBusId(bus, sizeof(bus), i);
    hipSetDevice(i);
    size_t freeb = 0, totb = 0;
    hipMemGetInfo(&freeb, &totb);
    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, i);
    printf("hip_dev %d  pci %s  free %7.1f GiB  total %7.1f GiB  cu %d  arch %s\n",
           i, bus, freeb / 1073741824.0, totb / 1073741824.0,
           p.multiProcessorCount, p.gcnArchName);
  }
  return 0;
}
