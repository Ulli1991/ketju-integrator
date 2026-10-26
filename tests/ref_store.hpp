// Ketju-integrator tests: reference data of the C library (KETJU), stored as text.
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
//
// The results of the C library (branch ketju-c) that the C++ tests compare against are in tests/reference_data/*.txt:
// one line per run "key count v0 v1 ..." with the doubles as hexadecimal floats (exact), the header lines (#) give
// the generating commit. Keys are names or a hash of the run's inputs.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace refstore {

struct Store {
    std::map<std::string, std::vector<double>> data;
    std::vector<std::string> order;
    std::string header;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream is(line);
            std::string key, tok;
            size_t n = 0;
            is >> key >> n;
            std::vector<double> v;
            v.reserve(n);
            while (is >> tok) v.push_back(std::strtod(tok.c_str(), nullptr));
            if (v.size() != n) { std::fprintf(stderr, "refstore: corrupt line %s in %s\n", key.c_str(), path.c_str()); std::abort(); }
            if (!data.count(key)) order.push_back(key);
            data[key] = std::move(v);
        }
        return true;
    }
    bool get(const std::string& key, std::vector<double>& out) const {
        auto it = data.find(key);
        if (it == data.end()) return false;
        out = it->second;
        return true;
    }
    void put(const std::string& key, const std::vector<double>& v) {
        if (!data.count(key)) order.push_back(key);
        data[key] = v;
    }
    void save(const std::string& path) const {
        std::FILE* f = std::fopen(path.c_str(), "w");
        if (!f) { std::fprintf(stderr, "refstore: cannot write %s\n", path.c_str()); std::abort(); }
        std::fputs(header.c_str(), f);
        for (const auto& k : order) {
            const auto& v = data.at(k);
            std::fprintf(f, "%s %zu", k.c_str(), v.size());
            for (double d : v) std::fprintf(f, " %a", d);
            std::fputc('\n', f);
        }
        std::fclose(f);
    }
};

// FNV-1a hash of doubles / ints (bit patterns)
struct Hash {
    std::uint64_t h = 1469598103934665603ull;
    void bytes(const void* p, size_t n) {
        const unsigned char* c = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ull; }
    }
    void add(double d) { bytes(&d, sizeof d); }
    void add(long i) { bytes(&i, sizeof i); }
    void add(const std::vector<double>& v) { add((long)v.size()); for (double d : v) add(d); }
    std::string str() const { char b[24]; std::snprintf(b, sizeof b, "%016llx", (unsigned long long)h); return b; }
};

}  // namespace refstore
