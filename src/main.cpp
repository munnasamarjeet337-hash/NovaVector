#include "../httplib.h"
#include "../include/types.hpp"
#include "../include/metrics.hpp"
#include "../include/brute_force.hpp"
#include "../include/kd_tree.hpp"
#include "../include/hnsw.hpp"
#include "../include/ollama_client.hpp"

#include <iostream>
#include <mutex>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>

static const int DEMO_DIMS = 16;

std::string escapeJson(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else o += c;
    }
    return o + '"';
}

std::string formatVectorJson(const std::vector<float>& v) {
    std::ostringstream ss; ss << '[';
    for (size_t i = 0; i < v.size(); i++) {
        if (i) ss << ',';
        ss << std::fixed << std::setprecision(4) << v[i];
    }
    return ss.str() + ']';
}

std::vector<float> parseVector(const std::string& s) {
    std::vector<float> v;
    std::istringstream ss(s); std::string t;
    while (std::getline(ss, t, ',')) {
        try { v.push_back(std::stof(t)); } catch (...) {}
    }
    return v;
}

std::string extractJsonString(const std::string& body, const std::string& key) {
    size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) return "";
    p = body.find(':', p) + 1;
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) p++;
    if (p >= body.size() || body[p] != '"') return "";
    p++;
    std::string res;
    while (p < body.size()) {
        if (body[p] == '"') break;
        if (body[p] == '\\' && p + 1 < body.size()) {
            p++;
            if (body[p] == 'n') res += '\n';
            else res += body[p];
        } else res += body[p];
        p++;
    }
    return res;
}

int extractJsonInt(const std::string& body, const std::string& key, int def = 0) {
    size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) return def;
    p = body.find(':', p) + 1;
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) p++;
    try { return std::stoi(body.substr(p)); } catch (...) { return def; }
}

void applyCors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

std::vector<std::string> chunkContent(const std::string& text, int chunkWords = 250, int overlap = 30) {
    std::istringstream ss(text);
    std::vector<std::string> words;
    std::string w;
    while (ss >> w) words.push_back(w);
    if (words.empty()) return {};
    if ((int)words.size() <= chunkWords) return {text};

    std::vector<std::string> chunks;
    int step = chunkWords - overlap;
    for (int i = 0; i < (int)words.size(); i += step) {
        int end = std::min(i + chunkWords, (int)words.size());
        std::string chunk;
        for (int j = i; j < end; j++) {
            if (j > i) chunk += ' ';
            chunk += words[j];
        }
        chunks.push_back(chunk);
        if (end == (int)words.size()) break;
    }
    return chunks;
}

class DocumentDatabase {
    std::unordered_map<int, DocumentRecord> store_;
    HNSWIndex hnsw_;
    FlatIndex fallback_flat_;
    std::mutex mu_;
    int next_id_ = 1;
    int dims_ = 0;

public:
    DocumentDatabase() : hnsw_(16, 200) {}

    int insert(const std::string& title, const std::string& text, const std::vector<float>& emb) {
        std::lock_guard<std::mutex> lk(mu_);
        if (dims_ == 0) dims_ = (int)emb.size();
        DocumentRecord doc{next_id_++, title, text, emb};
        store_[doc.id] = doc;
        VectorRecord vr{doc.id, title, "doc", emb};
        hnsw_.insert(vr, Metrics::cosine);
        fallback_flat_.insert(vr);
        return doc.id;
    }

    std::vector<std::pair<float, DocumentRecord>> search(const std::vector<float>& q, int k, float max_dist = 0.7f) {
        std::lock_guard<std::mutex> lk(mu_);
        if (store_.empty()) return {};
        auto raw = (store_.size() < 10)
            ? fallback_flat_.search(q, k, Metrics::cosine)
            : hnsw_.search(q, k, 50, Metrics::cosine);
        std::vector<std::pair<float, DocumentRecord>> out;
        for (auto& [d, id] : raw) {
            if (store_.count(id) && d <= max_dist) out.push_back({d, store_[id]});
        }
        return out;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!store_.count(id)) return false;
        store_.erase(id);
        hnsw_.remove(id);
        fallback_flat_.remove(id);
        return true;
    }

    std::vector<DocumentRecord> all() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<DocumentRecord> r;
        for (auto& [_, v] : store_) r.push_back(v);
        return r;
    }

    size_t size() { std::lock_guard<std::mutex> lk(mu_); return store_.size(); }
    int getDims() { return dims_; }
};

class CoreVectorEngine {
    std::unordered_map<int, VectorRecord> store_;
    FlatIndex bf_;
    SpatialKDTree kdt_;
    HNSWIndex hnsw_;
    std::mutex mu_;
    int next_id_ = 1;

public:
    explicit CoreVectorEngine(int d) : kdt_(d), hnsw_(16, 200) {}

    int insert(const std::string& meta, const std::string& cat, const std::vector<float>& emb, DistanceFunction dist) {
        std::lock_guard<std::mutex> lk(mu_);
        VectorRecord v{next_id_++, meta, cat, emb};
        store_[v.id] = v;
        bf_.insert(v); kdt_.insert(v); hnsw_.insert(v, dist);
        return v.id;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!store_.count(id)) return false;
        store_.erase(id); bf_.remove(id); hnsw_.remove(id);
        std::vector<VectorRecord> rem;
        for (auto& [_, v] : store_) rem.push_back(v);
        kdt_.rebuild(rem);
        return true;
    }

    struct SearchResult {
        std::vector<VectorRecord> hits;
        std::vector<float> dists;
        long long latency_us;
        std::string algo, metric;
    };

    SearchResult search(const std::vector<float>& q, int k, const std::string& metric, const std::string& algo) {
        std::lock_guard<std::mutex> lk(mu_);
        auto dfn = Metrics::resolve(metric);
        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<float, int>> raw;
        if (algo == "bruteforce") raw = bf_.search(q, k, dfn);
        else if (algo == "kdtree") raw = kdt_.search(q, k, dfn);
        else raw = hnsw_.search(q, k, 50, dfn);

        long long us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();

        SearchResult out; out.latency_us = us; out.algo = algo; out.metric = metric;
        for (auto& [d, id] : raw) {
            if (store_.count(id)) {
                out.hits.push_back(store_[id]);
                out.dists.push_back(d);
            }
        }
        return out;
    }

    struct BenchResult { long long bfUs, kdUs, hnswUs; int count; };
    BenchResult benchmark(const std::vector<float>& q, int k, const std::string& metric) {
        std::lock_guard<std::mutex> lk(mu_);
        auto dfn = Metrics::resolve(metric);
        auto time_it = [&](auto fn) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t0).count();
        };
        return {
            time_it([&]{ bf_.search(q, k, dfn); }),
            time_it([&]{ kdt_.search(q, k, dfn); }),
            time_it([&]{ hnsw_.search(q, k, 50, dfn); }),
            (int)store_.size()
        };
    }

    std::vector<VectorRecord> all() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<VectorRecord> r;
        for (auto& [_, v] : store_) r.push_back(v);
        return r;
    }

    HNSWIndex::GraphMetrics graphInfo() {
        std::lock_guard<std::mutex> lk(mu_);
        return hnsw_.inspectTopology();
    }

    size_t size() { std::lock_guard<std::mutex> lk(mu_); return store_.size(); }
};

void initDemoCorpus(CoreVectorEngine& db) {
    auto dist = Metrics::resolve("cosine");
    db.insert("Linked List: nodes connected by pointers", "cs", {0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}, dist);
    db.insert("Binary Search Tree: O(log n) search and insert", "cs", {0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}, dist);
    db.insert("Calculus: derivatives integrals and limits", "math", {0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}, dist);
    db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food", {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}, dist);
    db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports", {0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}, dist);
}

int main() {
    CoreVectorEngine engine(DEMO_DIMS);
    DocumentDatabase docDB;
    OllamaBridge ollama;

    initDemoCorpus(engine);
    httplib::Server svr;

    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        applyCors(res); res.status = 204;
    });

    svr.Get("/search", [&](const httplib::Request& req, httplib::Response& res) {
        applyCors(res);
        auto q = parseVector(req.get_param_value("v"));
        if ((int)q.size() != DEMO_DIMS) {
            res.set_content("{\"error\":\"need 16D vector\"}", "application/json");
            return;
        }
        int k = 5; try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        std::string metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        std::string algo = req.get_param_value("algo"); if (algo.empty()) algo = "hnsw";

        auto out = engine.search(q, k, metric, algo);
        std::ostringstream ss;
        ss << "{\"results\":[";
        for (size_t i = 0; i < out.hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":" << out.hits[i].id
               << ",\"metadata\":" << escapeJson(out.hits[i].metadata)
               << ",\"category\":" << escapeJson(out.hits[i].category)
               << ",\"distance\":" << std::fixed << std::setprecision(6) << out.dists[i]
               << ",\"embedding\":" << formatVectorJson(out.hits[i].values) << '}';
        }
        ss << "],\"latencyUs\":" << out.latency_us
           << ",\"algo\":" << escapeJson(out.algo)
           << ",\"metric\":" << escapeJson(out.metric) << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/items", [&](const httplib::Request&, httplib::Response& res) {
        applyCors(res);
        auto items = engine.all();
        std::ostringstream ss; ss << '[';
        for (size_t i = 0; i < items.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":" << items[i].id
               << ",\"metadata\":" << escapeJson(items[i].metadata)
               << ",\"category\":" << escapeJson(items[i].category)
               << ",\"embedding\":" << formatVectorJson(items[i].values) << '}';
        }
        ss << ']';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/benchmark", [&](const httplib::Request& req, httplib::Response& res) {
        applyCors(res);
        auto q = parseVector(req.get_param_value("v"));
        if ((int)q.size() != DEMO_DIMS) {
            res.set_content("{\"error\":\"need 16D vector\"}", "application/json"); return;
        }
        int k = 5; try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        std::string metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto b = engine.benchmark(q, k, metric);
        std::ostringstream ss;
        ss << "{\"bruteforceUs\":" << b.bfUs << ",\"kdtreeUs\":" << b.kdUs
           << ",\"hnswUs\":" << b.hnswUs << ",\"itemCount\":" << b.count << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/hnsw-info", [&](const httplib::Request&, httplib::Response& res) {
        applyCors(res);
        auto gi = engine.graphInfo();
        std::ostringstream ss;
        ss << "{\"topLayer\":" << gi.max_level << ",\"nodeCount\":" << gi.total_nodes << ",\"nodesPerLayer\":[";
        for (size_t i = 0; i < gi.level_node_distribution.size(); i++) {
            if (i) ss << ','; ss << gi.level_node_distribution[i];
        }
        ss << "],\"edgesPerLayer\":[";
        for (size_t i = 0; i < gi.level_edge_distribution.size(); i++) {
            if (i) ss << ','; ss << gi.level_edge_distribution[i];
        }
        ss << "],\"nodes\":[";
        for (size_t i = 0; i < gi.nodes.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":" << gi.nodes[i].id << ",\"metadata\":" << escapeJson(gi.nodes[i].label)
               << ",\"category\":" << escapeJson(gi.nodes[i].tag) << ",\"maxLyr\":" << gi.nodes[i].max_lvl << '}';
        }
        ss << "],\"edges\":[";
        for (size_t i = 0; i < gi.edges.size(); i++) {
            if (i) ss << ',';
            ss << "{\"src\":" << gi.edges[i].origin << ",\"dst\":" << gi.edges[i].target << ",\"lyr\":" << gi.edges[i].level << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/doc/insert", [&](const httplib::Request& req, httplib::Response& res) {
        applyCors(res);
        std::string title = extractJsonString(req.body, "title");
        std::string text  = extractJsonString(req.body, "text");
        if (title.empty() || text.empty()) {
            res.set_content("{\"error\":\"need title and text\"}", "application/json"); return;
        }
        auto chunks = chunkContent(text, 250, 30);
        std::vector<int> ids;
        for (int i = 0; i < (int)chunks.size(); i++) {
            auto emb = ollama.generateEmbedding(chunks[i]);
            if (emb.empty()) {
                res.set_content("{\"error\":\"Ollama unavailable. Run: ollama serve\"}", "application/json");
                return;
            }
            std::string ct = (chunks.size() > 1) ? title + " [" + std::to_string(i+1) + "/" + std::to_string(chunks.size()) + "]" : title;
            ids.push_back(docDB.insert(ct, chunks[i], emb));
        }
        std::ostringstream ss;
        ss << "{\"ids\":[";
        for (size_t i = 0; i < ids.size(); i++) { if (i) ss << ','; ss << ids[i]; }
        ss << "],\"chunks\":" << chunks.size() << ",\"dims\":" << docDB.getDims() << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/doc/list", [&](const httplib::Request&, httplib::Response& res) {
        applyCors(res);
        auto docs = docDB.all();
        std::ostringstream ss; ss << '[';
        for (size_t i = 0; i < docs.size(); i++) {
            if (i) ss << ',';
            std::string preview = docs[i].text.substr(0, 120);
            if (docs[i].text.size() > 120) preview += "...";
            ss << "{\"id\":" << docs[i].id
               << ",\"title\":" << escapeJson(docs[i].title)
               << ",\"preview\":" << escapeJson(preview)
               << ",\"words\":" << (int)std::count(docs[i].text.begin(), docs[i].text.end(), ' ') + 1 << '}';
        }
        ss << ']';
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/doc/search", [&](const httplib::Request& req, httplib::Response& res) {
        applyCors(res);
        auto qText = extractJsonString(req.body, "question");
        int k = extractJsonInt(req.body, "k", 3);
        auto emb = ollama.generateEmbedding(qText);
        auto hits = docDB.search(emb, k);
        std::ostringstream ss; ss << "{\"contexts\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":" << hits[i].second.id << ",\"title\":" << escapeJson(hits[i].second.title)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/doc/ask", [&](const httplib::Request& req, httplib::Response& res) {
        applyCors(res);
        auto qText = extractJsonString(req.body, "question");
        int k = extractJsonInt(req.body, "k", 3);
        auto emb = ollama.generateEmbedding(qText);
        if (emb.empty()) {
            res.set_content("{\"error\":\"Ollama embedding generation failed.\"}", "application/json"); return;
        }
        auto hits = docDB.search(emb, k);
        std::ostringstream ctx;
        for (size_t i = 0; i < hits.size(); i++) {
            ctx << "[" << (i+1) << "] " << hits[i].second.title << ":\n" << hits[i].second.text << "\n\n";
        }
        std::string prompt = "Context:\n" + ctx.str() + "\nQuestion: " + qText + "\n\nAnswer directly and clearly:";
        auto answer = ollama.complete(prompt);

        std::ostringstream ss;
        ss << "{\"answer\":" << escapeJson(answer)
           << ",\"model\":" << escapeJson(ollama.generation_model)
           << ",\"contexts\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":" << hits[i].second.id << ",\"title\":" << escapeJson(hits[i].second.title)
               << ",\"text\":" << escapeJson(hits[i].second.text)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "],\"docCount\":" << docDB.size() << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        applyCors(res);
        bool up = ollama.isHealthy();
        std::ostringstream ss;
        ss << "{\"ollamaAvailable\":" << (up ? "true" : "false")
           << ",\"embedModel\":" << escapeJson(ollama.embedding_model)
           << ",\"genModel\":" << escapeJson(ollama.generation_model)
           << ",\"docCount\":" << docDB.size()
           << ",\"docDims\":" << docDB.getDims()
           << ",\"demoDims\":" << DEMO_DIMS
           << ",\"demoCount\":" << engine.size() << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream file("index.html");
        if (!file.is_open()) { res.status = 404; return; }
        res.set_content(std::string(std::istreambuf_iterator<char>(file),
                                    std::istreambuf_iterator<char>()), "text/html");
    });

    std::cout << "NovaVector Engine running on http://localhost:8080" << std::endl;
    svr.listen("0.0.0.0", 8080);
    return 0;
}