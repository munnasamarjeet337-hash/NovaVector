#pragma once
#include <vector>
#include <string>
#include <functional>

struct VectorRecord {
    int id;
    std::string metadata;
    std::string category;
    std::vector<float> values;
};

struct DocumentRecord {
    int id;
    std::string title;
    std::string text;
    std::vector<float> values;
};

using DistanceFunction = std::function<float(const std::vector<float>&, const std::vector<float>&)>;