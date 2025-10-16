#include "linear_model_cpp.hpp"
#include <algorithm>
#include <numeric>

LinearModelIndex::LinearModelIndex() : a(0.0), b(0.0) {}

void LinearModelIndex::build(const std::vector<double>& keys) {
    data = keys;
    if (keys.empty()) return;

    size_t n = keys.size();
    std::vector<double> positions(n);
    std::iota(positions.begin(), positions.end(), 0.0);

    double mean_x = std::accumulate(keys.begin(), keys.end(), 0.0) / n;
    double mean_y = std::accumulate(positions.begin(), positions.end(), 0.0) / n;

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        num += (keys[i] - mean_x) * (positions[i] - mean_y);
        den += (keys[i] - mean_x) * (keys[i] - mean_x);
    }

    if (den != 0.0) {
        a = num / den;
        b = mean_y - a * mean_x;
    } else {
        a = 0.0;
        b = n / 2.0;
    }
}

std::pair<bool, int> LinearModelIndex::search(double key, int window) const {
    if (data.empty()) return {false, 0};

    int n = static_cast<int>(data.size());
    int pred = static_cast<int>(a * key + b);
    pred = std::max(0, std::min(n - 1, pred));

    int left = std::max(0, pred - window);
    int right = std::min(n, pred + window);

    auto it = std::lower_bound(data.begin() + left, data.begin() + right, key);
    bool found = (it != data.begin() + right && *it == key);
    return {found, 1};
}

size_t LinearModelIndex::getMemoryUsage() const {
    return data.size() * sizeof(double) + sizeof(a) + sizeof(b);
}
