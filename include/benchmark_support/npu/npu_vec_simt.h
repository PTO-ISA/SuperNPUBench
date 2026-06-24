#include <common/pto_tileop.hpp>

#define DEFAULT_HASH_SEED 0
#define c1 0xcc9e2d51
#define c2 0x1b873593
#define c3 0xe6546b64
#define rot_c1 15
#define rot_c2 13

#define USE_FDIV

#ifdef USE_FDIV
#define DIV_U32(r_e_s_u_l_t, d_i_v_i_d_e_n_d, d_i_v_i_s_o_r, m_a_g_i_c, s_h_i_f_t) \
    do \
    { \
        uint32_t q =__umulhi(d_i_v_i_d_e_n_d, m_a_g_i_c); \
        uint32_t t_e_m_p = ((d_i_v_i_d_e_n_d - q) >> 1) + q; \
        r_e_s_u_l_t = t_e_m_p >> (s_h_i_f_t-1); \
    } while(0)

#define MOD_U32(r_e_s_u_l_t, d_i_v_i_d_e_n_d, d_i_v_i_s_o_r, m_a_g_i_c, s_h_i_f_t) \
    do \
    { \
        uint32_t q =__umulhi(d_i_v_i_d_e_n_d, m_a_g_i_c); \
        uint32_t t_e_m_p = ((d_i_v_i_d_e_n_d - q) >> 1) + q; \
        r_e_s_u_l_t = d_i_v_i_d_e_n_d - d_i_v_i_s_o_r * (uint32_t)(t_e_m_p >> (s_h_i_f_t - 1));\
    } while(0)

#else
    #define DIV_U32(r_e_s_u_l_t, d_i_v_i_d_e_n_d, d_i_v_i_s_o_r, m_a_g_i_c, s_h_i_f_t) \
    do \
    { \
        r_e_s_u_l_t = d_i_v_i_d_e_n_d / d_i_v_i_s_o_r; \
    } while(0)

    #define MOD_U32(r_e_s_u_l_t, d_i_v_i_d_e_n_d, d_i_v_i_s_o_r, m_a_g_i_c, s_h_i_f_t) \
    do \
    { \
        r_e_s_u_l_t = d_i_v_i_d_e_n_d % d_i_v_i_s_o_r;\
    } while(0)

#endif

template<int thread_num, int block_num>
void __mtc__ hashtable_insert(
    int8_t*  slot,
    int32_t* keys,
    int32_t* values,
    uint32_t capacity,
    uint32_t dummy2,
    int num,
    uint32_t dummy3,
    uint32_t capacity_magic, int dummy0,
    uint32_t capacity_shift, int dummy1
)
{
    uint16_t thread_idx = blkv_get_index_x();
    uint16_t block_idx = blkv_get_index_y();

    int32_t key_sentinel = int32_t(-1);
    //int entry_size = sizeof(int32_t) + sizeof(uint32_t) * 4;
    const int entry_size = 16; // alignas(16)
    /*accel::parallel_for(accel::dim3{1024}, [=]() LAUNCH_BOUND(1024)*/
    {
        // main loop start
        for (int tid = thread_idx + thread_num * block_idx;
          tid < num; tid += thread_num * block_num) {
            int32_t key = keys[tid];
            int32_t value = values[tid];

            int len = 8;
            int nblocks = 2;

            int32_t* blocks = (int32_t*)(keys + tid);

            uint32_t h = DEFAULT_HASH_SEED;
            for (int j = 0; j < 2; j++) {
                uint32_t k1 = blocks[j];
                k1 *= c1;
                k1 = (k1 << rot_c1) | (k1 >> (32 - rot_c1));
                k1 *= c2;
                h ^= k1;
                h = (h << rot_c2) | (h >> (32 - rot_c2));
                h = h * 5 + c3;
            }

            // h = compute_remaining_bytes(data, len, tail_offset, h);
            // Finalize hash.
            h ^= len;
            h ^= h >> 16;
            h *= 0x85ebca6b;
            h ^= h >> 13;
            h *= 0xc2b2ae35;
            h ^= h >> 16;
            uint32_t b = h;

            int32_t curr_slot = b & (capacity - 1);
            // int32_t curr_slot = b % capacity;
            // uint32_t curr_slot = 0;
            // MOD_U32(curr_slot, b, capacity, capacity_magic, capacity_shift);
            // MOD_U32(curr_slot, b, (2550000U), (2769502708u), (22u));

            uint32_t count = 0;
            while (true) {
              if (count >= capacity)
                break;
              int32_t* key_addr = (int32_t*)(slot + entry_size * curr_slot);
              int32_t curr_key = *key_addr;
              if (curr_key == key_sentinel) {
                // slot is empty
                // int32_t old = atomicCAS(key_addr, key_sentinel, key);
                int32_t old = *key_addr;
                *key_addr = key;
                if (old == key_sentinel) {
                  int32_t* value_addr = (int32_t*)(slot + entry_size * curr_slot + sizeof(int32_t));
                  value_addr[0] = value;
                  break;
                }
              }
              count++;
              curr_slot = (curr_slot + 1) & (capacity - 1);
            //   curr_slot = (curr_slot + 1) % capacity;
            // curr_slot += 1;
            //     MOD_U32(curr_slot, curr_slot, capacity, capacity_magic, capacity_shift);
                // MOD_U32(curr_slot, curr_slot, (2550000U), (2769502708U), (22u));
            }

        }
        // main loop end
    }
    //);
}

// template <uint32_t MaxThreadsPerBlock>
// __simt_vf__ LAUNCH_BOUND(MaxThreadsPerBlock)[aicore]
template<int thread_num, int block_num>
void __mtc__ callee(uint8_t* slot,
  int32_t* keys,
  int32_t* values_output,
  uint32_t capacity,
  int num,
  uint32_t capacity_magic,
  uint32_t capacity_shift,
  int32_t key_sentinel,
  int32_t value_sentinel, const int entry_size)
{
    uint16_t thread_idx = blkv_get_index_x();
    uint16_t block_idx = blkv_get_index_y();
    // main loop start
    for (int block_base_idx = thread_num * block_idx;
         block_base_idx < num; block_base_idx += thread_num * block_num)
    {
        int tid = block_base_idx + thread_idx;
        if (tid < num)
        {
            int32_t key = keys[tid];

            int len = 8;
            int nblocks = 2;

            uint32_t* blocks = (uint32_t*)(keys + tid);

            uint32_t h = DEFAULT_HASH_SEED;
            for (int j = 0; j < 2; j++)
            {
                uint32_t k1 = blocks[j];
                k1 *= c1;
                k1 = (k1 << rot_c1) | (k1 >> (32 - rot_c1));
                k1 *= c2;
                h ^= k1;
                h = (h << rot_c2) | (h >> (32 - rot_c2));
                h = h * 5 + c3;
            }

            // h = compute_remaining_bytes(data, len, tail_offset, h);
            // Finalize hash.
            h ^= len;
            h ^= h >> 16;
            h *= 0x85ebca6b;
            h ^= h >> 13;
            h *= 0xc2b2ae35;
            h ^= h >> 16;
            uint32_t b = h;

            int32_t curr_slot = b & (capacity - 1);
            // int32_t curr_slot = b % capacity;
            // uint32_t curr_slot = 0;
            // MOD_U32(curr_slot, b, capacity, capacity_magic, capacity_shift);

            while (true)
            {
                int32_t* key_addr = (int32_t*)(slot + entry_size * curr_slot);
                int32_t curr_key = *key_addr;
                if (curr_key == key_sentinel)
                {
                    // key not found
                    values_output[tid] = value_sentinel;
                    break;
                }

                if (curr_key == key)
                {
                    // found key
                    int32_t* value_addr = (int32_t*)(slot + entry_size * curr_slot + sizeof(int32_t));
                    values_output[tid] = value_addr[0];
                    break;
                }

                curr_slot = (curr_slot + 1) & (capacity - 1);
                // curr_slot = (curr_slot + 1) % capacity;
                // curr_slot += 1;
                // MOD_U32(curr_slot, curr_slot, capacity, capacity_magic, capacity_shift);
            }
        }  // tid < num
    }
    // main loop end
}

void hashtable_lookup(
    uint8_t* slot,
    int32_t* keys,
    int32_t* values_output,
    uint32_t capacity, uint32_t dummy2,
    int num, uint32_t dummy3,
    uint32_t capacity_magic, int dummy0,
    uint32_t capacity_shift, int dummy1)
{
    int32_t key_sentinel = int32_t(-1);
    int32_t value_sentinel = int32_t(-1);
    // __ubuf__ int32_t* write_buffer = (__ubuf__ int32_t*)get_imm(0x0);
    const int entry_size = 16;

    constexpr int BLOCK_THREADS = 1024;
    // accel::async_invoke<callee<BLOCK_THREADS>>(accel::dim3{BLOCK_THREADS, 1, 1}, slot,
    //                                          keys, values_output, capacity, num, capacity_magic, capacity_shift,
    //                                          key_sentinel, value_sentinel, entry_size);
    callee<BLOCK_THREADS, 1><<<BLOCK_THREADS, 1, 1>>>(slot, keys, values_output, capacity, num, capacity_magic, capacity_shift, key_sentinel, value_sentinel, entry_size);
}