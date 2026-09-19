#pragma once
#include "types.hpp"
#include <cmath>
#include <string>

namespace Metrics {
    inline float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    inline float cosine(const std::vector<float>& a, const std::vector<float>& b) {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a < 1e-9f || norm_b < 1e-9f) return 1.0f;
        return 1.0f - (dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    inline float manhattan(const std::vector<float>& a, const std::vector<float>& b) {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += std::abs(a[i] - b[i]);
        }
        return sum;
    }

    inline DistanceFunction resolve(const std::string& name) {
        if (name == "cosine") return cosine;
        if (name == "manhattan") return manhattan;
        return euclidean;
    }
}