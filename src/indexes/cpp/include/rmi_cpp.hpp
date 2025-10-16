#pragma once
#include <vector>
#include <utility>

/**
 * @brief Two-stage Recursive Model Index (C++ version).
 * Stage 0 — root linear model predicts global position.
 * Stage 1 — each leaf segment fits its own linear model for refinement.
 */
class RecursiveModelIndex {
public:
    explicit RecursiveModelIndex(int fanout = 128);

    void build(const std::vector<double>& keys);
    std::pair<bool, int> search(double key, int safety = 8) const;
    size_t getMemoryUsage() const;

private:
    int fanout;
    int n;
    double a0, b0;                        // root model
    std::vector<double> seg_a, seg_b;     // leaf models
    std::vector<int> seg_start, seg_end;  // segment bounds
    std::vector<int> seg_err;             // error windows
    std::vector<double> data;             // sorted keys

    void allocSegments(int f);
};
