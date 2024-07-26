#include "ipbytebin.h"
#include<sys/resource.h>

EXPORT uint32_t ip_byte_bin(uint64_t *q, uint64_t *d, uint32_t B){
    uint64_t ret = 0;
    uint32_t sub_ret;
    uint32_t size = B / 64;
    for(int i = 0; i < 4; i++){
        sub_ret = 0;
        for(int j = 0; j < size; j++){
            sub_ret += __builtin_popcountll(d[j] & (*q));
            q++;
        }
        ret += (sub_ret << i);
    }
    return ret;
}

