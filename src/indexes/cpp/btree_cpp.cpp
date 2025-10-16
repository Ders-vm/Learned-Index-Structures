#include "btree_cpp.hpp"
#include <cmath>
#include <memory>

BTreeNode::BTreeNode(bool isLeaf) : leaf(isLeaf) {}

bool BTreeNode::isFull(int order) const {
    return static_cast<int>(keys.size()) >= order - 1;
}

BTree::BTree(int order_) : order(order_), size(0) {
    root = std::make_shared<BTreeNode>(true);
}

void BTree::build(const std::vector<double>& sortedKeys) {
    size = static_cast<int>(sortedKeys.size());
    if (size == 0) return;

    // Create leaf level
    std::vector<std::shared_ptr<BTreeNode>> leafNodes;
    int keysPerLeaf = order - 1;

    for (int i = 0; i < size; i += keysPerLeaf) {
        auto node = std::make_shared<BTreeNode>(true);
        int end = std::min(i + keysPerLeaf, size);
        node->keys.assign(sortedKeys.begin() + i, sortedKeys.begin() + end);
        leafNodes.push_back(node);
    }

    // Build internal levels
    auto current = leafNodes;
    while (current.size() > 1) {
        std::vector<std::shared_ptr<BTreeNode>> nextLevel;
        for (size_t i = 0; i < current.size(); i += order) {
            auto parent = std::make_shared<BTreeNode>(false);
            size_t groupEnd = std::min(i + order, current.size());
            for (size_t j = i; j < groupEnd; ++j) {
                parent->children.push_back(current[j]);
                if (j < groupEnd - 1) {
                    parent->keys.push_back(current[j + 1]->keys.front());
                }
            }
            nextLevel.push_back(parent);
        }
        current = nextLevel;
    }

    root = current.front();
}

std::pair<bool, int> BTree::search(double key) const {
    return searchRecursive(root, key);
}

std::pair<bool, int> BTree::searchRecursive(std::shared_ptr<BTreeNode> node, double key) const {
    int idx = std::lower_bound(node->keys.begin(), node->keys.end(), key) - node->keys.begin();

    if (idx < static_cast<int>(node->keys.size()) && node->keys[idx] == key)
        return {true, 1};

    if (node->leaf)
        return {false, 1};

    return searchRecursive(node->children[idx], key);
}

size_t BTree::getMemoryUsage() const {
    return getMemoryRecursive(root);
}

size_t BTree::getMemoryRecursive(std::shared_ptr<BTreeNode> node) const {
    size_t mem = node->keys.size() * sizeof(double) + node->children.size() * sizeof(void*) + sizeof(BTreeNode);
    if (!node->leaf) {
        for (const auto& child : node->children)
            mem += getMemoryRecursive(child);
    }
    return mem;
}
