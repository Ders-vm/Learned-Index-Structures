#include "rmi_cpp.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

RecursiveModelIndex::RecursiveModelIndex(int fanout_)
    : fanout(fanout_), n(0), a0(0.0), b0(0.0) {}

void RecursiveModelIndex::allocSegments(int f) {
    seg_a.assign(f, 0.0);
    seg_b.assign(f, 0.0);
    seg_start.assign(f, 0);
    seg_end.assign(f, 0);
    seg_err.assign(f, 0);
}

void RecursiveModelIndex::build(const std::vector<double>& keys) {
    data = keys;
    n = static_cast<int>(data.size());
    if (n == 0) { allocSegments(fanout); return; }

    std::vector<double> pos(n);
    std::iota(pos.begin(), pos.end(), 0.0);

    double mean_x = std::accumulate(data.begin(), data.end(), 0.0) / n;
    double mean_y = std::accumulate(pos.begin(), pos.end(), 0.0) / n;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        num += (data[i] - mean_x) * (pos[i] - mean_y);
        den += (data[i] - mean_x) * (data[i] - mean_x);
    }
    a0 = den ? num / den : 0.0;
    b0 = mean_y - a0 * mean_x;

    allocSegments(fanout);
    for (int s = 0; s < fanout; ++s) {
        int start = static_cast<int>(std::floor(s * n / (double)fanout));
        int end   = static_cast<int>(std::floor((s + 1) * n / (double)fanout));
        seg_start[s] = start;
        seg_end[s]   = end;

        if (end <= start) continue;
        const auto beginIt = data.begin() + start;
        const auto endIt   = data.begin() + end;

        std::vector<double> kseg(beginIt, endIt);
        std::vector<double> pseg(end - start);
        std::iota(pseg.begin(), pseg.end(), start);

        double mean_ks = std::accumulate(kseg.begin(), kseg.end(), 0.0) / kseg.size();
        double mean_ps = std::accumulate(pseg.begin(), pseg.end(), 0.0) / pseg.size();

        num = den = 0.0;
        for (size_t i = 0; i < kseg.size(); ++i) {
            num += (kseg[i] - mean_ks) * (pseg[i] - mean_ps);
            den += (kseg[i] - mean_ks) * (kseg[i] - mean_ks);
        }
        double a = den ? num / den : 0.0;
        double b = mean_ps - a * mean_ks;
        seg_a[s] = a;
        seg_b[s] = b;

        // compute max error
        int max_err = 0;
        for (size_t i = 0; i < kseg.size(); ++i) {
            int pred = static_cast<int>(a * kseg[i] + b);
            int err  = std::abs(pred - static_cast<int>(pseg[i]));
            if (err > max_err) max_err = err;
        }
        seg_err[s] = std::max(2, max_err);
    }
}

std::pair<bool, int> RecursiveModelIndex::search(double key, int safety) const {
    if (n == 0) return {false, 1};

    double pos0 = a0 * key + b0;
    int s = std::clamp(static_cast<int>((pos0 / (n - 1)) * fanout), 0, fanout - 1);

    double a = seg_a[s];
    double b = seg_b[s];
    int pred = static_cast<int>(std::round(a * key + b));
    int w = seg_err[s] + safety;
    int left = std::max(seg_start[s], pred - w);
    int right = std::min(seg_end[s], pred + w + 1);

    auto it = std::lower_bound(data.begin() + left, data.begin() + right, key);
    bool found = (it != data.begin() + right && *it == key);
    return {found, 1};
}

size_t RecursiveModelIndex::getMemoryUsage() const {
    return data.size() * sizeof(double)
        + seg_a.size() * sizeof(double)
        + seg_b.size() * sizeof(double)
        + seg_start.size() * sizeof(int)
        + seg_end.size() * sizeof(int)
        + seg_err.size() * sizeof(int)
        + sizeof(a0) + sizeof(b0);
}
