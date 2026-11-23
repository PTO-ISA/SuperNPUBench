// Byte and selection boundaries that random whole-layer fixtures rarely hit.
#include "deepseekv4.1/layer.hpp"
#include <string.h>

using namespace deepseek_v41;

static bool fp4_boundaries() {
    const float group[] = {.25f, -.75f, 1.25f, -1.75f,        2.5f,          -3.5f, 5.f, -.125f,
                           -0.f, -.25f, 2.75f, 0x1.bffffep0f, 0x1.c00002p0f, -5.f,  6.f, -6.f};
    float x[32];
    for (int i = 0; i < 16; ++i) {
        x[i] = group[i];
        x[16 + i] = -2.f * group[i];
    }
    // The two KV groups have BF16 scales 1 and 2. The index row has one UE8M0 scale 2.
    const uint8_t lower[] = {0x90, 0xb2, 0xd4, 0x86, 0x88, 0x35, 0xe4, 0xf7,
                             0x18, 0x3a, 0x5c, 0x0e, 0x00, 0xbd, 0x6c, 0x7f};
    const uint8_t even[] = {0xa0, 0xc2, 0xe4, 0x86, 0x88, 0x35, 0xe4, 0xf7,
                            0x28, 0x4a, 0x6c, 0x0e, 0x00, 0xbd, 0x6c, 0x7f};
    const uint8_t key[] = {0x90, 0xa1, 0xc2, 0x84, 0x88, 0x23, 0xc2, 0xd5,
                           0x28, 0x4a, 0x6c, 0x0e, 0x00, 0xbd, 0x6c, 0x7f};
    const uint8_t query[] = {0x90, 0xa1, 0xc3, 0x85, 0x88, 0x23, 0xd2, 0xd5,
                             0x29, 0x4b, 0x6d, 0x0f, 0x10, 0xbd, 0x7c, 0x7f};
    CompressedRow c2, c1;
    IndexRow k, q;
    quant4<CompGroup, Midpoint::Lower>(x, c2.data, c2.scale);
    quant4<CompGroup, Midpoint::Even>(x, c1.data, c1.scale);
    quant4<MxGroup, Midpoint::Even>(x, k.data, k.scale);
    quant4<MxGroup, Midpoint::Upper>(x, q.data, q.scale);
    return c1.scale[0] == 0x3f80 && c1.scale[1] == 0x4000 &&
           memcmp(c1.scale, c2.scale, sizeof(c1.scale)) == 0 && k.scale[0] == 128 && q.scale[0] == 128 &&
           memcmp(c2.data, lower, sizeof(lower)) == 0 && memcmp(c1.data, even, sizeof(even)) == 0 &&
           memcmp(k.data, key, sizeof(key)) == 0 && memcmp(q.data, query, sizeof(query)) == 0;
}

static bool fp8_boundaries() {
    const float lanes[] = {448, -448, 0.f,     -0.f,     0x1p-9f, -0x1p-9f, 0x1p-10f, -0x1p-10f,
                           240, -240, 1.0625f, -1.0625f, 1.1875f, -1.1875f, 0x1p-11f, -0x1p-11f};
    const uint8_t codes[] = {0x7e, 0xfe, 0,    0x80, 1,    0x81, 0, 0x80,
                             0x77, 0xf7, 0x38, 0xb8, 0x3a, 0xba, 0, 0x80};
    float x[2 * MxGroup];
    for (int i = 0; i < MxGroup; ++i) {
        x[i] = lanes[i % 16];
        x[i + MxGroup] = 3.f * x[i];
    }
    auto a = activation(x);
    WindowRow window;
    quant8<MxGroup>(x, window.data, window.scale);
    return a.scale[0] == 127 && a.scale[1] == 129 && window.scale[0] == 0x3f80 && a.data[MxGroup] == 0x7a &&
           memcmp(a.data, codes, sizeof(codes)) == 0 && memcmp(a.data, window.data, MxGroup) == 0;
}

static bool selection_boundaries() {
    float scores[8] = {-INFINITY, 2, 2, -INFINITY, -INFINITY, 10, 20, 30};
    int selected[4];
    const int expected[] = {1, 2, -1, -1};
    select_rows(scores, 5, selected);
    if (memcmp(selected, expected, sizeof(expected)) != 0)
        return false;
    select_rows(scores, 0, selected);
    for (int id : selected)
        if (id != -1)
            return false;

    static Weights w;
    float x[Hidden] = {1.f};
    for (auto &row : w.router)
        row[0] = -60.f;
    Route routes[TopK];
    route(x, w, routes);
    for (int i = 0; i < TopK; ++i)
        if (routes[i].expert != i || routes[i].slot != i || !(fabsf(routes[i].weight - .25f) <= 1.e-6f))
            return false;
    return true;
}

int main() {
    if (get_thread_idx() != 0)
        return 0;
    if (!fp4_boundaries())
        return 1;
    if (!fp8_boundaries())
        return 2;
    if (!selection_boundaries())
        return 3;
    return 0;
}
