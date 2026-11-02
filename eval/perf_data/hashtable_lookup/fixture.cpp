#include <cstdint>
#include "multi_thread_res_check.h"
#include "basic_op/hashtable_lookup/hashtable_lookup.hpp"
using supernpu::hash_lookup::TableEntry;
extern "C" const TableEntry table[2550000];
extern "C" const int64_t queries[6144];
extern "C" const int32_t golden[6144];
asm(".pushsection .rodata\n.balign 4096\n.global table\ntable:\n.incbin \"/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/hashtable_lookup/data/table_data_70pct.bin\"\n.popsection\n");
asm(".pushsection .rodata\n.balign 4096\n.global queries\nqueries:\n.incbin \"/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/hashtable_lookup/data/input_keys_data.bin\"\n.popsection\n");
asm(".pushsection .rodata\n.balign 4096\n.global golden\ngolden:\n.incbin \"/Users/blacktraker/Programming/gitproj/DV4/SuperNPUBench/benchmark/one-level-arch/test/kernel/hashtable_lookup/data/lookup_golden.bin\"\n.popsection\n");
alignas(4096) int32_t output[6146];
MultiThreadResCheckSync sync{};
volatile unsigned errors;
volatile unsigned checked;
extern "C" int main() {
 unsigned tid=get_thread_idx();
 if(tid==0){output[0]=0x12345678;output[6145]=0x12345678;}
 res_check_publish_inputs(sync,tid);
 supernpu::multi_thread::hashtable_lookup<2550000,4>(table,queries,output+1,6144);
 res_check_wait_for_all(sync,tid);
 if(tid==0){
  unsigned n=0;
  for(unsigned i=0;i<6144;++i) if(output[i+1]!=golden[i]) ++n;
  if(output[0]!=0x12345678 || output[6145]!=0x12345678) ++n;
  errors=n;
  __asm__ volatile("":::"memory");
  checked=1;
 } else { while(checked==0){} }
 __asm__ volatile("":::"memory");
 unsigned n=errors;
 *reinterpret_cast<volatile uint32_t*>(0x10009000ULL)=n?((n<<16)|0x3333):0x5555;
 return 0;
}
