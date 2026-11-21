#include "wse/deepseek_v41.hpp"
#include "fixture.hpp"
#ifdef V41_MSGB_RUNTIME
#include "wse/msgb_sdma.hpp"
#endif

using namespace deepseek_v41;
extern "C" {
extern test::Input v41_input;
alignas(64) test::Result v41_result;
}
static wse::Sram sram;
static ExpertInbox shard_inboxes[2];

template <class Sdma> static int run_layer(Sdma &sdma) {
    wse::Memory mem(sram, sdma, shard_inboxes[0], shard_inboxes[1]);
    if (!mem.stage_weights(v41_input.shards))
        return 1;
    int status = test::run(v41_input, v41_result, mem);
    v41_result.dynamic_puts = mem.puts;
    v41_result.dynamic_gets = mem.gets;
    v41_result.sdma_transfers = mem.completed;
    v41_result.sdma_bytes = mem.copied_bytes;
    return status;
}

#ifdef V41_MSGB_RUNTIME
// The resident runtime provisions the endpoints and passes each worker its peer.
extern "C" uint64_t v41_layer_client(uint64_t service_endpoint) {
    wse::MessageSdma sdma(service_endpoint);
    return run_layer(sdma);
}

extern "C" uint64_t v41_copy_service(uint64_t client_endpoint) {
    wse::SingleCoreSdma engine(v41_input.weights, sram, v41_input.shards);
    for (int i = 0; i < v41_input.shards + test::Steps; ++i)
        if (!wse::serve_one_copy(client_endpoint, engine))
            return 1;
    return 0;
}
#else
int main() {
    if (get_thread_idx() != 0)
        return 0;
    wse::SingleCoreSdma sdma(v41_input.weights, sram, v41_input.shards);
    return run_layer(sdma);
}
#endif
