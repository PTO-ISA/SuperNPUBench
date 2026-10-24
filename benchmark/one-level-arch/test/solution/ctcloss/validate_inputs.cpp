#include "solution/ctcloss/ctcloss_v2_i32.hpp"
#include <cassert>
#include <fstream>
#include <string>
#include <vector>
template<class T> std::vector<T> read(const std::string &path,size_t count) {
  std::vector<T> out(count); std::ifstream in(path,std::ios::binary);
  assert(in && in.read(reinterpret_cast<char *>(out.data()),count*sizeof(T)));
  assert(in.peek()==std::ifstream::traits_type::eof()); return out;
}
int main(int argc,char **argv) {
  using namespace ctcloss_v2;
  assert(argc==2);
    const std::string dir=argv[1];
    ctc_loss_forward::Arguments core{};
    auto x=read<float>(dir+"/log_probs.bin",195*256*41);
    auto y=read<int32_t>(dir+"/targets.bin",256*74);
    auto lengths=read<int32_t>(dir+"/input_lengths.bin",256);
    auto targets=read<int32_t>(dir+"/target_lengths.bin",256);
    std::vector<float> al(256*195*149),nll(256);
    std::vector<uint32_t> off(256);
    PaddedI32Arguments network{x.data(),y.data(),lengths.data(),targets.data(),
      al.data(),nll.data(),195,256,41,74,0,Reduction::none,false};
    assert(prepare(network,x.size(),y.size(),lengths.size(),targets.size(),
      al.size(),nll.size(),off.data(),off.size(),core)==InputError::ok);
    assert(core.max_states==149 && core.lp_time_stride==10496);
    assert(core.lp_batch_stride==41 && core.alpha_batch_stride==29055);
    assert(core.alpha_time_stride==149 && off[255]==255*74);
    for (int b=0;b<256;++b) {
      assert(lengths[b]==195 && targets[b]==(b==0?40:74));
    }
}
