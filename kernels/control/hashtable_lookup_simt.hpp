#ifndef HASHTABLE_LOOKUP_SIMT_HPP
#define HASHTABLE_LOOKUP_SIMT_HPP

#include <common/pto_tileop.hpp>
#include <cstdint>

// ============================================================================
// MurmurHash3 constants
// ============================================================================

#define DEFAULT_HASH_SEED 0u
#define c1 0xcc9e2d51u
#define c2 0x1b873593u
#define c3 0xe6546b64u
#define rot_c1 15u
#define rot_c2 13u

// ============================================================================
// hashfind — SIMT linear-probing hash table lookup kernel (no probe buffers)
//   Single kernel launch using 3D grid: <<<thread_num, 1, 1>>>
//   Each thread processes multiple keys with stride = thread_num
// ============================================================================

template <int32_t thread_num>
void __mtc__ hashfind(uint8_t* __restrict__ slot,
                      int64_t* __restrict__ keys,
                      int32_t* __restrict__ values_output,
                      uint32_t capacity,
                      int32_t num,
                      int32_t entry_size,
                      int32_t max_probe,
                      int64_t key_sentinel,
                      int32_t value_sentinel)
{
    int32_t tid = blkv_get_index_x();

    // main loop: each thread processes multiple keys with stride = thread_num
    for (int32_t idx = tid; idx < num; idx += thread_num)
    {
        int64_t key = keys[idx];

        uint32_t len = 8u;
        uint32_t* blocks = (uint32_t*)(keys + idx);

        uint32_t h = DEFAULT_HASH_SEED;
        for (int32_t j = 0; j < 2; j++)
        {
            uint32_t k1 = blocks[j];
            k1 *= c1;
            k1 = (k1 << rot_c1) | (k1 >> (32u - rot_c1));
            k1 *= c2;
            h ^= k1;
            h = (h << rot_c2) | (h >> (32u - rot_c2));
            h = h * 5u + c3;
        }

        // Finalize hash.
        h ^= len;
        h ^= h >> 16u;
        h *= 0x85ebca6bu;
        h ^= h >> 13;
        h *= 0xc2b2ae35u;
        h ^= h >> 16u;

        uint32_t curr_slot = h % capacity;

        for (int probe_cnt = 0; probe_cnt < max_probe; probe_cnt++) {
            int64_t* key_addr = (int64_t*)(slot + entry_size * curr_slot);
            int64_t curr_key = *key_addr;

            if (curr_key == key_sentinel) {
                // key not found
                values_output[idx] = value_sentinel;
                probe_cnt = max_probe;
            }

            if (curr_key == key) {
                // found key
                int32_t* value_addr = (int32_t*)(slot + entry_size * curr_slot + sizeof(int64_t));
                values_output[idx] = value_addr[0];
                probe_cnt = max_probe;
            }

            curr_slot = (curr_slot + 1u) % capacity;
        }
    }
}

#endif // HASHTABLE_LOOKUP_SIMT_HPP
