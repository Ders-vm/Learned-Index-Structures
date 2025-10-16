#pragma once
#include <vector>
#include <utility>
#include <cmath>

/**
 * @brief Simple Linear Regression Learned Index (C++ baseline)
 * Mimics Python's LearnedIndex: predicts key position via linear regression,
 * then corrects locally with binary search.
 */
class LinearModelIndex {
public:
    LinearModelIndex();

    void build(const std::vector<double>& keys);
    std::pair<bool, int> search(double key, int window = 64) const;
    size_t getMemoryUsage() const;

private:
    double a; // slope
    double b; // intercept
    std::vector<double> data;
};
