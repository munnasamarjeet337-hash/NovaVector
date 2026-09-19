#pragma once
#include "types.hpp"
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>

struct KDNode {
    VectorRecord record;
    KDNode* left = nullptr;
    KDNode* right = nullptr;
    explicit KDNode(const VectorRecord& r) : record(r) {}
};

class SpatialKDTree {
    KDNode* root_ = nullptr;
    int dimensions_;

    void clear(KDNode* node) {
        if (!node) return;
        clear(node->left);
        clear(node->right);
        delete node;
    }

    KDNode* insertRecursive(KDNode* node, const VectorRecord& item, int depth) {
        if (!node) return new KDNode(item);
        int axis = depth % dimensions_;
        if (item.values[axis] < node->record.values[axis]) {
            node->left = insertRecursive(node->left, item, depth + 1);
        } else {
            node->right = insertRecursive(node->right, item, depth + 1);
        }
        return node;
    }

    void searchRecursive(KDNode* node, const std::vector<float>& target, int k, int depth,
                         DistanceFunction metric, std::priority_queue<std::pair<float, int>>& best) {
        if (!node) return;
        float dist = metric(target, node->record.values);
        if ((int)best.size() < k || dist < best.top().first) {
            best.push({dist, node->record.id});
            if ((int)best.size() > k) best.pop();
        }

        int axis = depth % dimensions_;
        float axis_diff = target[axis] - node->record.values[axis];
        KDNode* near_branch = axis_diff < 0 ? node->left : node->right;
        KDNode* far_branch  = axis_diff < 0 ? node->right : node->left;

        searchRecursive(near_branch, target, k, depth + 1, metric, best);
        if ((int)best.size() < k || std::abs(axis_diff) < best.top().first) {
            searchRecursive(far_branch, target, k, depth + 1, metric, best);
        }
    }

public:
    explicit SpatialKDTree(int dim) : dimensions_(dim) {}
    ~SpatialKDTree() { clear(root_); }

    void insert(const VectorRecord& item) {
        root_ = insertRecursive(root_, item, 0);
    }

    std::vector<std::pair<float, int>> search(const std::vector<float>& target, int k, DistanceFunction metric) {
        std::priority_queue<std::pair<float, int>> heap;
        searchRecursive(root_, target, k, 0, metric, heap);
        std::vector<std::pair<float, int>> out;
        while (!heap.empty()) {
            out.push_back(heap.top());
            heap.pop();
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    void rebuild(const std::vector<VectorRecord>& items) {
        clear(root_);
        root_ = nullptr;
        for (const auto& item : items) insert(item);
    }
};