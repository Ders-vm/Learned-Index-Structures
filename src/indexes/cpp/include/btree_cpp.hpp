#pragma once
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>

/**
 * @brief Simple B-Tree baseline used for comparison with learned index structures.
 * 
 * Supports:
 *  - Bulk build from sorted keys
 *  - Recursive search
 *  - Approximate memory usage
 */
class BTreeNode {
public:
    bool leaf;
    std::vector<double> keys;
    std::vector<std::shared_ptr<BTreeNode>> children;

    explicit BTreeNode(bool isLeaf);
    bool isFull(int order) const;
};

class BTree {
public:
    explicit BTree(int order = 128);

    void build(const std::vector<double>& sortedKeys);
    std::pair<bool, int> search(double key) const;
    size_t getMemoryUsage() const;

private:
    int order;
    std::shared_ptr<BTreeNode> root;
    int size;

    std::pair<bool, int> searchRecursive(std::shared_ptr<BTreeNode> node, double key) const;
    size_t getMemoryRecursive(std::shared_ptr<BTreeNode> node) const;
};
