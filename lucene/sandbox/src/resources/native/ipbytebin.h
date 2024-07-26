
#include <stdint.h>


#ifdef _MSC_VER
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__) && !defined(__clang__)
#define EXPORT __attribute__((externally_visible,visibility("default")))
#elif __clang__
#define EXPORT __attribute__((visibility("default")))
#endif


EXPORT uint32_t ip_byte_bin(uint64_t *q, uint64_t *d, uint32_t B);
