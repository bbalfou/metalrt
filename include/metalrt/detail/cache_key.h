#pragma once
#include <metalrt/types.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

namespace metalrt {
namespace detail {
// Canonicalize and validate public option data before using it in any cache.
inline KernelCompileOptions validated_compile_options(KernelCompileOptions options) {
    std::sort(options.constants.begin(), options.constants.end(),
              [](const auto& a, const auto& b) { return a.index < b.index; });
    uint32_t previous = 0;
    bool first = true;
    for (const auto& c : options.constants) {
        const bool boolean = c.type == FunctionConstant::Type::Bool;
        if (c.type != FunctionConstant::Type::Bool && c.type != FunctionConstant::Type::Int32 &&
            c.type != FunctionConstant::Type::UInt32 && c.type != FunctionConstant::Type::Float32)
            throw std::invalid_argument("unknown function constant type");
        if (c.index > 65535 || c.value.size() != (boolean ? sizeof(bool) : 4) ||
            (!first && c.index == previous))
            throw std::invalid_argument("invalid or duplicate function constant");
        if (boolean && c.value[0] > 1)
            throw std::invalid_argument("invalid boolean function constant");
        previous = c.index;
        first = false;
    }
    return options;
}
inline std::string compile_options_key(const KernelCompileOptions& options) {
    std::string key = options.fast_math ? "fast=1;" : "fast=0;";
    for (const auto& c : options.constants) {
        key += std::to_string(c.index) + ":" + std::to_string(static_cast<int>(c.type)) + ":";
        for (auto byte : c.value) {
            constexpr char hex[] = "0123456789abcdef";
            key += hex[byte >> 4];
            key += hex[byte & 15];
        }
        key += ";";
    }
    return key;
}
// Stable SHA-256 cache names, independent of std::hash implementation, process,
// host endianness and C++ runtime. Metal additionally verifies that a loaded
// archive contains the exact requested pipeline descriptor and function.
inline std::string kernel_cache_digest(const std::string& bytes) {
    constexpr uint32_t constants[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    std::array<uint32_t, 8> state{{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f,
                                   0x9b05688c, 0x1f83d9ab, 0x5be0cd19}};
    const auto rotate = [](uint32_t value, unsigned count) {
        return (value >> count) | (value << (32 - count));
    };
    const auto compress = [&](const uint8_t* block) {
        uint32_t words[64];
        for (unsigned i = 0; i < 16; ++i)
            words[i] = (uint32_t(block[4 * i]) << 24) | (uint32_t(block[4 * i + 1]) << 16) |
                       (uint32_t(block[4 * i + 2]) << 8) | uint32_t(block[4 * i + 3]);
        for (unsigned i = 16; i < 64; ++i) {
            const uint32_t a = words[i - 15], b = words[i - 2];
            words[i] = words[i - 16] + (rotate(a, 7) ^ rotate(a, 18) ^ (a >> 3)) + words[i - 7] +
                       (rotate(b, 17) ^ rotate(b, 19) ^ (b >> 10));
        }
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4], f = state[5],
                 g = state[6], h = state[7];
        for (unsigned i = 0; i < 64; ++i) {
            const uint32_t first = h + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) +
                                   ((e & f) ^ (~e & g)) + constants[i] + words[i];
            const uint32_t second =
                (rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d + first;
            d = c;
            c = b;
            b = a;
            a = first + second;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    };
    size_t offset = 0;
    while (bytes.size() - offset >= 64) {
        compress(reinterpret_cast<const uint8_t*>(bytes.data() + offset));
        offset += 64;
    }
    std::array<uint8_t, 128> tail{};
    const size_t remaining = bytes.size() - offset;
    if (remaining)
        std::memcpy(tail.data(), bytes.data() + offset, remaining);
    tail[remaining] = 0x80;
    const size_t padding = remaining < 56 ? 64 : 128;
    const uint64_t bit_length = uint64_t(bytes.size()) * 8;
    for (unsigned i = 0; i < 8; ++i)
        tail[padding - 1 - i] = uint8_t(bit_length >> (8 * i));
    compress(tail.data());
    if (padding == 128)
        compress(tail.data() + 64);
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (uint32_t word : state)
        result << std::setw(8) << word;
    return result.str();
}
} // namespace detail
} // namespace metalrt
