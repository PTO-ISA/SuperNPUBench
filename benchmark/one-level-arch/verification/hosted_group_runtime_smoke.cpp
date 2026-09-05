#include <common/linx_group_runtime.h>

#include <stdint.h>

extern "C" int __linx_group_worker_main(uint32_t peId, void *opaque)
{
    (void)peId;
    (void)opaque;
    return 0;
}

int main()
{
    return linx_group_run(nullptr);
}
