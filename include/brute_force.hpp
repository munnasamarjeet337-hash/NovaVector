#pragma once
#include "types.hpp"
#include <vector>
#include <algorithm>

class FlatIndex {
public:
    std::vector<VectorRecord> records;

    void insert(const VectorRecord& item) { 
        records.push_back(item); 
    }

    std::vector<std::pair<float, int>> search(const std::vector<float>& query, int k, DistanceFunction metric) {
        std::vector<std::pair<float, int>> results;
        results.reserve(records.size());
        for (const auto& item : records) {
            results.push_back({metric(query, item.values), item.id});
        }
        std::sort(results.begin(), results.end());
        if ((int)results.size() > k) results.resize(k);
        return results;
    }

    void remove(int id) {
        records.erase(std::remove_if(records.begin(), records.end(),
            [id](const VectorRecord& item) { return item.id == id; }), records.end());
    }
};