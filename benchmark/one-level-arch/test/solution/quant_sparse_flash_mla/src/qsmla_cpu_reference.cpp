// Unified CPU golden generator for the FP16 BSND QSMLA test implementation.
//
// Two invocation styles are supported:
//   1. Read existing binary inputs and write one FP32 golden file.
//   2. --generate-deterministic: create FP16 Q/ORI-KV plus the SWA golden.
//
// All five logical source modes share one softmax row. Dot products, exp and
// P@V use double intermediates; the persisted golden is FP32. OpenMP only
// parallelizes independent (batch, q-token, q-head) rows.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    std::string mode;
    fs::path q_path, ori_path, cmp_path, ori_indices_path, cmp_indices_path;
    fs::path ori_lengths_path, cmp_lengths_path, output, output_dir;
    int b = 0, s1 = 0, n1 = 0, n2 = 0, d = 0, ori_s2 = 0, cmp_s2 = 0;
    int ori_topk = 0, cmp_topk = 0, cmp_ratio = 1;
    int win_left = -1, win_right = -1;
    double softmax_scale = 1.0;
    bool generate_deterministic = false;
};

static uint16_t float_to_half(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t mantissa = bits & 0x007fffffu;
    int exponent = static_cast<int>((bits >> 23) & 0xffu) - 112;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x00800000u) >> (1 - exponent);
        mantissa += 0xfffu + ((mantissa >> 13) & 1u);
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    mantissa += 0xfffu + ((mantissa >> 13) & 1u);
    if (mantissa & 0x00800000u) {
        mantissa = 0;
        if (++exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                                 (mantissa >> 13));
}

static float half_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x400u) == 0) { mantissa <<= 1; ++shift; }
            mantissa &= 0x3ffu;
            bits = sign | static_cast<uint32_t>(127 - 15 - shift) << 23 |
                   mantissa << 13;
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | mantissa << 13;
    } else {
        bits = sign | (exponent + 112u) << 23 | mantissa << 13;
    }
    return std::bit_cast<float>(bits);
}

template <class T>
static std::vector<T> read_exact(const fs::path& path, size_t count) {
    std::vector<T> values(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    if (!stream || static_cast<size_t>(stream.gcount()) != count * sizeof(T) ||
        stream.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("unexpected binary size: " + path.string());
    }
    return values;
}

template <class T>
static void write_all(const fs::path& path, const std::vector<T>& values) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(values.data()),
                 values.size() * sizeof(T));
    if (!stream) throw std::runtime_error("failed to write: " + path.string());
}

static std::map<std::string, std::string> parse_values(int argc, char** argv,
                                                       Options& options) {
    std::map<std::string, std::string> values;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--generate-deterministic") {
            options.generate_deterministic = true;
        } else if (key.rfind("--", 0) == 0 && i + 1 < argc) {
            values[key] = argv[++i];
        } else {
            throw std::runtime_error("invalid argument: " + key);
        }
    }
    return values;
}

static Options parse_options(int argc, char** argv) {
    Options o;
    auto v = parse_values(argc, argv, o);
    auto text = [&](const char* key, const std::string& fallback = {}) {
        auto it = v.find(key); return it == v.end() ? fallback : it->second;
    };
    auto integer = [&](const char* key, int fallback = 0) {
        auto value = text(key); return value.empty() ? fallback : std::stoi(value);
    };
    o.mode = text("--mode"); o.q_path = text("--q");
    o.ori_path = text("--ori-kv"); o.cmp_path = text("--cmp-kv");
    o.ori_indices_path = text("--ori-indices");
    o.cmp_indices_path = text("--cmp-indices");
    o.ori_lengths_path = text("--ori-lengths");
    o.cmp_lengths_path = text("--cmp-lengths");
    o.output = text("--output"); o.output_dir = text("--output-dir");
    o.b = integer("--b"); o.s1 = integer("--s1"); o.n1 = integer("--n1");
    o.n2 = integer("--n2"); o.d = integer("--d");
    o.ori_s2 = integer("--ori-s2"); o.cmp_s2 = integer("--cmp-s2");
    o.ori_topk = integer("--ori-topk"); o.cmp_topk = integer("--cmp-topk");
    o.cmp_ratio = integer("--cmp-ratio", 1);
    o.win_left = integer("--win-left", -1);
    o.win_right = integer("--win-right", -1);
    auto scale = text("--softmax-scale");
    if (!scale.empty()) o.softmax_scale = std::stod(scale);
    const std::vector<std::string> modes = {
        "SWA", "HCA", "CSA", "ORI_SPARSE", "ORI_CMP_SPARSE"};
    if (std::find(modes.begin(), modes.end(), o.mode) == modes.end())
        throw std::runtime_error("unsupported --mode");
    if (o.b <= 0 || o.s1 <= 0 || o.n1 <= 0 || o.n2 <= 0 || o.d <= 0 ||
        o.ori_s2 <= 0 || o.n1 % o.n2 != 0)
        throw std::runtime_error("invalid BSND shape");
    if (o.cmp_ratio <= 0) throw std::runtime_error("cmp-ratio must be positive");
    return o;
}

static size_t q_offset(const Options& o, int b, int s, int n, int d) {
    return ((static_cast<size_t>(b) * o.s1 + s) * o.n1 + n) * o.d + d;
}

static size_t kv_offset(int s2, int n2, int d_size,
                        int b, int s, int n, int d) {
    return ((static_cast<size_t>(b) * s2 + s) * n2 + n) * d_size + d;
}

static int floor_div(int numerator, int denominator) {
    return numerator >= 0 ? numerator / denominator
                          : -((-numerator + denominator - 1) / denominator);
}

struct Entry { bool cmp; int token; };

static std::vector<int> collect_indices(const std::vector<int32_t>& candidates,
                                        const std::vector<int32_t>& lengths,
                                        size_t list_offset, size_t length_offset,
                                        int topk, int source_s2, int causal_end,
                                        bool use_length) {
    int count = use_length ? std::clamp(lengths.at(length_offset), 0, topk) : topk;
    std::vector<int> selected;
    for (int i = 0; i < count; ++i) {
        int index = candidates.at(list_offset + i);
        if (index == -1) break;
        if (index >= 0 && index < source_s2 && index < causal_end)
            selected.push_back(index);
    }
    return selected;
}

static std::vector<float> calculate(const Options& o,
                                    const std::vector<uint16_t>& q,
                                    const std::vector<uint16_t>& ori,
                                    const std::vector<uint16_t>& cmp,
                                    const std::vector<int32_t>& ori_indices,
                                    const std::vector<int32_t>& cmp_indices,
                                    const std::vector<int32_t>& ori_lengths,
                                    const std::vector<int32_t>& cmp_lengths) {
    std::vector<float> output(static_cast<size_t>(o.b) * o.s1 * o.n1 * o.d);
    const int group = o.n1 / o.n2;
#pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < o.b * o.s1 * o.n1; ++row) {
        int q_head = row % o.n1;
        int q_pos = (row / o.n1) % o.s1;
        int batch = row / (o.n1 * o.s1);
        int kv_head = q_head / group;
        std::vector<Entry> entries;
        const bool indexed_ori = o.mode == "ORI_SPARSE" || o.mode == "ORI_CMP_SPARSE";
        if (indexed_ori) {
            size_t list = ((static_cast<size_t>(batch) * o.s1 + q_pos) * o.n2 + kv_head) * o.ori_topk;
            size_t length = (static_cast<size_t>(batch) * o.s1 + q_pos) * o.n2 + kv_head;
            int end = std::clamp(o.ori_s2 - o.s1 + q_pos + 1, 0, o.ori_s2);
            for (int token : collect_indices(ori_indices, ori_lengths, list, length,
                                             o.ori_topk, o.ori_s2, end, true))
                entries.push_back({false, token});
        } else {
            int diagonal = o.ori_s2 - o.s1 + q_pos;
            int begin = o.win_left < 0 ? 0 : diagonal - o.win_left;
            int end = o.win_right < 0 ? o.ori_s2 : diagonal + o.win_right + 1;
            begin = std::clamp(begin, 0, o.ori_s2);
            end = std::clamp(end, begin, o.ori_s2);
            for (int token = begin; token < end; ++token) entries.push_back({false, token});
        }
        const bool has_cmp = o.mode == "HCA" || o.mode == "CSA" || o.mode == "ORI_CMP_SPARSE";
        if (has_cmp) {
            int end = std::clamp(floor_div(o.cmp_s2 * o.cmp_ratio - o.s1 + q_pos + 1,
                                           o.cmp_ratio), 0, o.cmp_s2);
            if (o.mode == "HCA") {
                for (int token = 0; token < end; ++token) entries.push_back({true, token});
            } else {
                size_t list = ((static_cast<size_t>(batch) * o.s1 + q_pos) * o.n2 + kv_head) * o.cmp_topk;
                size_t length = (static_cast<size_t>(batch) * o.s1 + q_pos) * o.n2 + kv_head;
                bool use_length = o.mode == "ORI_CMP_SPARSE" && !cmp_lengths.empty();
                for (int token : collect_indices(cmp_indices, cmp_lengths, list, length,
                                                 o.cmp_topk, o.cmp_s2, end, use_length))
                    entries.push_back({true, token});
            }
        }
        if (entries.empty()) continue;
        std::vector<double> weights(entries.size());
        double row_max = -std::numeric_limits<double>::infinity();
        for (size_t e = 0; e < entries.size(); ++e) {
            double dot = 0.0;
            for (int d = 0; d < o.d; ++d) {
                size_t qo = q_offset(o, batch, q_pos, q_head, d);
                size_t ko = entries[e].cmp
                    ? kv_offset(o.cmp_s2, o.n2, o.d, batch, entries[e].token, kv_head, d)
                    : kv_offset(o.ori_s2, o.n2, o.d, batch, entries[e].token, kv_head, d);
                dot += half_to_float(q[qo]) *
                       half_to_float(entries[e].cmp ? cmp[ko] : ori[ko]);
            }
            weights[e] = dot * o.softmax_scale;
            row_max = std::max(row_max, weights[e]);
        }
        double denominator = 0.0;
        for (double& weight : weights) { weight = std::exp(weight - row_max); denominator += weight; }
        for (int d = 0; d < o.d; ++d) {
            double value = 0.0;
            for (size_t e = 0; e < entries.size(); ++e) {
                size_t ko = entries[e].cmp
                    ? kv_offset(o.cmp_s2, o.n2, o.d, batch, entries[e].token, kv_head, d)
                    : kv_offset(o.ori_s2, o.n2, o.d, batch, entries[e].token, kv_head, d);
                value += weights[e] * half_to_float(entries[e].cmp ? cmp[ko] : ori[ko]);
            }
            output[q_offset(o, batch, q_pos, q_head, d)] = static_cast<float>(value / denominator);
        }
    }
    return output;
}

int main(int argc, char** argv) {
    try {
        Options o = parse_options(argc, argv);
        const size_t q_count = static_cast<size_t>(o.b) * o.s1 * o.n1 * o.d;
        const size_t ori_count = static_cast<size_t>(o.b) * o.ori_s2 * o.n2 * o.d;
        std::vector<uint16_t> q, ori;
        if (o.generate_deterministic) {
            if (o.mode != "SWA" || o.output_dir.empty())
                throw std::runtime_error("deterministic generation requires SWA and --output-dir");
            q.resize(q_count); ori.resize(ori_count);
            for (size_t i = 0; i < q.size(); ++i)
                q[i] = float_to_half(static_cast<float>((i * 31 + 17) % 100) / 100.0f - 0.5f);
            for (size_t i = 0; i < ori.size(); ++i)
                ori[i] = float_to_half(static_cast<float>((i * 31 + 34) % 100) / 100.0f - 0.5f);
            fs::create_directories(o.output_dir);
            write_all(o.output_dir / "q.fp16.bin", q);
            write_all(o.output_dir / "ori_kv.fp16.bin", ori);
            o.output = o.output_dir / "golden.fp32.bin";
        } else {
            if (o.q_path.empty() || o.ori_path.empty() || o.output.empty())
                throw std::runtime_error("--q, --ori-kv and --output are required");
            q = read_exact<uint16_t>(o.q_path, q_count);
            ori = read_exact<uint16_t>(o.ori_path, ori_count);
        }
        const bool has_cmp = o.mode == "HCA" || o.mode == "CSA" || o.mode == "ORI_CMP_SPARSE";
        std::vector<uint16_t> cmp;
        if (has_cmp) cmp = read_exact<uint16_t>(o.cmp_path,
            static_cast<size_t>(o.b) * o.cmp_s2 * o.n2 * o.d);
        std::vector<int32_t> ori_indices, cmp_indices, ori_lengths, cmp_lengths;
        if (o.mode == "ORI_SPARSE" || o.mode == "ORI_CMP_SPARSE") {
            ori_indices = read_exact<int32_t>(o.ori_indices_path,
                static_cast<size_t>(o.b) * o.s1 * o.n2 * o.ori_topk);
            ori_lengths = read_exact<int32_t>(o.ori_lengths_path,
                static_cast<size_t>(o.b) * o.s1 * o.n2);
        }
        if (o.mode == "CSA" || o.mode == "ORI_CMP_SPARSE") {
            cmp_indices = read_exact<int32_t>(o.cmp_indices_path,
                static_cast<size_t>(o.b) * o.s1 * o.n2 * o.cmp_topk);
            if (!o.cmp_lengths_path.empty()) cmp_lengths = read_exact<int32_t>(
                o.cmp_lengths_path, static_cast<size_t>(o.b) * o.s1 * o.n2);
        }
        auto golden = calculate(o, q, ori, cmp, ori_indices, cmp_indices,
                                ori_lengths, cmp_lengths);
        write_all(o.output, golden);
        if (o.generate_deterministic) {
            std::ofstream manifest(o.output_dir / "manifest.txt");
            manifest << "mode=" << o.mode << "\nB=" << o.b << " S1=" << o.s1
                     << " ORI_S2=" << o.ori_s2 << " N1=" << o.n1
                     << " N2=" << o.n2 << " D=" << o.d << "\n";
        }
        std::cout << "golden=" << o.output << " elements=" << golden.size() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qsmla_cpu_reference: " << error.what() << "\n";
        return 2;
    }
}
