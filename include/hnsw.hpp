#pragma once
#include "types.hpp"
#include <unordered_map>
#include <queue>
#include <random>
#include <cmath>
#include <algorithm>

class HNSWIndex {
public:
    struct GraphMetrics {
        int max_level;
        int total_nodes;
        std::vector<int> level_node_distribution;
        std::vector<int> level_edge_distribution;
        struct NodeEntry { int id; std::string label; std::string tag; int max_lvl; };
        struct EdgeEntry { int origin; int target; int level; };
        std::vector<NodeEntry> nodes;
        std::vector<EdgeEntry> edges;
    };

private:
    struct GraphNode {
        VectorRecord record;
        int max_layer;
        std::vector<std::vector<int>> neighbors;
    };

    std::unordered_map<int, GraphNode> graph_;
    int max_edges_;
    int max_edges_base_;
    int ef_construction_;
    float level_mult_;
    int top_level_ = -1;
    int entry_point_ = -1;
    std::mt19937 prng_;

    int generateRandomLevel() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return static_cast<int>(std::floor(-std::log(dist(prng_)) * level_mult_));
    }

    std::vector<std::pair<float, int>> traverseLayer(
        const std::vector<float>& query, int ep, int ef, int level, DistanceFunction metric)
    {
        std::unordered_map<int, bool> visited;
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<>> frontier;
        std::priority_queue<std::pair<float, int>> candidates;

        float d0 = metric(query, graph_[ep].record.values);
        visited[ep] = true;
        frontier.push({d0, ep});
        candidates.push({d0, ep});

        while (!frontier.empty()) {
            auto [curr_dist, curr_id] = frontier.top();
            frontier.pop();

            if ((int)candidates.size() >= ef && curr_dist > candidates.top().first) break;
            if (level >= (int)graph_[curr_id].neighbors.size()) continue;

            for (int neighbor_id : graph_[curr_id].neighbors[level]) {
                if (visited[neighbor_id] || !graph_.count(neighbor_id)) continue;
                visited[neighbor_id] = true;
                float nd = metric(query, graph_[neighbor_id].record.values);

                if ((int)candidates.size() < ef || nd < candidates.top().first) {
                    frontier.push({nd, neighbor_id});
                    candidates.push({nd, neighbor_id});
                    if ((int)candidates.size() > ef) candidates.pop();
                }
            }
        }

        std::vector<std::pair<float, int>> result;
        while (!candidates.empty()) {
            result.push_back(candidates.top());
            candidates.pop();
        }
        std::sort(result.begin(), result.end());
        return result;
    }

    std::vector<int> selectNeighbors(std::vector<std::pair<float, int>>& candidates, int limit) {
        std::vector<int> out;
        int count = std::min((int)candidates.size(), limit);
        for (int i = 0; i < count; ++i) out.push_back(candidates[i].second);
        return out;
    }

public:
    explicit HNSWIndex(int m = 16, int ef_build = 200)
        : max_edges_(m), max_edges_base_(2 * m), ef_construction_(ef_build),
          level_mult_(1.0f / std::log((float)m)), prng_(42) {}

    void insert(const VectorRecord& item, DistanceFunction metric) {
        int id = item.id;
        int level = generateRandomLevel();
        graph_[id] = {item, level, std::vector<std::vector<int>>(level + 1)};

        if (entry_point_ == -1) {
            entry_point_ = id;
            top_level_ = level;
            return;
        }

        int curr_obj = entry_point_;
        for (int lc = top_level_; lc > level; --lc) {
            if (lc < (int)graph_[curr_obj].neighbors.size()) {
                auto path = traverseLayer(item.values, curr_obj, 1, lc, metric);
                if (!path.empty()) curr_obj = path[0].second;
            }
        }

        for (int lc = std::min(top_level_, level); lc >= 0; --lc) {
            auto path = traverseLayer(item.values, curr_obj, ef_construction_, lc, metric);
            int m_bound = (lc == 0) ? max_edges_base_ : max_edges_;
            auto chosen = selectNeighbors(path, m_bound);
            graph_[id].neighbors[lc] = chosen;

            for (int nid : chosen) {
                if (!graph_.count(nid)) continue;
                if ((int)graph_[nid].neighbors.size() <= lc) graph_[nid].neighbors.resize(lc + 1);
                auto& edges = graph_[nid].neighbors[lc];
                edges.push_back(id);

                if ((int)edges.size() > m_bound) {
                    std::vector<std::pair<float, int>> dists;
                    for (int cand : edges) {
                        if (graph_.count(cand)) {
                            dists.push_back({metric(graph_[nid].record.values, graph_[cand].record.values), cand});
                        }
                    }
                    std::sort(dists.begin(), dists.end());
                    edges.clear();
                    for (int i = 0; i < m_bound && i < (int)dists.size(); ++i) {
                        edges.push_back(dists[i].second);
                    }
                }
            }
            if (!path.empty()) curr_obj = path[0].second;
        }

        if (level > top_level_) {
            top_level_ = level;
            entry_point_ = id;
        }
    }

    std::vector<std::pair<float, int>> search(const std::vector<float>& query, int k, int ef, DistanceFunction metric) {
        if (entry_point_ == -1) return {};
        int curr_obj = entry_point_;
        for (int lc = top_level_; lc > 0; --lc) {
            if (lc < (int)graph_[curr_obj].neighbors.size()) {
                auto path = traverseLayer(query, curr_obj, 1, lc, metric);
                if (!path.empty()) curr_obj = path[0].second;
            }
        }
        auto results = traverseLayer(query, curr_obj, std::max(ef, k), 0, metric);
        if ((int)results.size() > k) results.resize(k);
        return results;
    }

    void remove(int id) {
        if (!graph_.count(id)) return;
        for (auto& [_, node] : graph_) {
            for (auto& layer : node.neighbors) {
                layer.erase(std::remove(layer.begin(), layer.end(), id), layer.end());
            }
        }
        if (entry_point_ == id) {
            entry_point_ = -1;
            for (auto& [nid, _] : graph_) {
                if (nid != id) { entry_point_ = nid; break; }
            }
        }
        graph_.erase(id);
    }

    GraphMetrics inspectTopology() {
        GraphMetrics info;
        info.max_level = top_level_;
        info.total_nodes = (int)graph_.size();
        int max_l = std::max(top_level_ + 1, 1);
        info.level_node_distribution.assign(max_l, 0);
        info.level_edge_distribution.assign(max_l, 0);

        for (auto& [nid, node] : graph_) {
            info.nodes.push_back({nid, node.record.metadata, node.record.category, node.max_layer});
            for (int lc = 0; lc <= node.max_layer && lc < max_l; ++lc) {
                info.level_node_distribution[lc]++;
                if (lc < (int)node.neighbors.size()) {
                    for (int neighbor : node.neighbors[lc]) {
                        if (nid < neighbor) {
                            info.level_edge_distribution[lc]++;
                            info.edges.push_back({nid, neighbor, lc});
                        }
                    }
                }
            }
        }
        return info;
    }

    size_t size() const { return graph_.size(); }
};