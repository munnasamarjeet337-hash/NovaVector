#pragma once
#include "../httplib.h"
#include <string>
#include <vector>
#include <sstream>

class OllamaBridge {
    std::string server_host_;
    int server_port_;

    std::string escapeJsonString(const std::string& input) {
        std::string buffer;
        for (char c : input) {
            if (c == '"') buffer += "\\\"";
            else if (c == '\\') buffer += "\\\\";
            else if (c == '\n') buffer += "\\n";
            else if (c == '\r') buffer += "\\r";
            else if (c == '\t') buffer += "\\t";
            else buffer += c;
        }
        return buffer;
    }

public:
    std::string embedding_model = "nomic-embed-text";
    std::string generation_model = "llama3.2";

    OllamaBridge(std::string host = "127.0.0.1", int port = 11434)
        : server_host_(std::move(host)), server_port_(port) {}

    bool isHealthy() {
        httplib::Client client(server_host_, server_port_);
        client.set_connection_timeout(2, 0);
        auto response = client.Get("/api/tags");
        return response && response->status == 200;
    }

    std::vector<float> generateEmbedding(const std::string& text) {
        httplib::Client client(server_host_, server_port_);
        client.set_connection_timeout(3, 0);
        client.set_read_timeout(30, 0);

        std::string payload = "{\"model\":\"" + embedding_model + "\",\"prompt\":\"" + escapeJsonString(text) + "\"}";
        auto response = client.Post("/api/embeddings", payload, "application/json");
        if (!response || response->status != 200) return {};

        size_t start = response->body.find("\"embedding\"");
        if (start == std::string::npos) return {};
        start = response->body.find('[', start);
        if (start == std::string::npos) return {};

        size_t end = start + 1, depth = 1;
        while (end < response->body.size() && depth > 0) {
            if (response->body[end] == '[') depth++;
            else if (response->body[end] == ']') depth--;
            end++;
        }

        std::string raw_array = response->body.substr(start + 1, end - start - 2);
        std::vector<float> output;
        std::istringstream stream(raw_array);
        std::string token;
        while (std::getline(stream, token, ',')) {
            try { output.push_back(std::stof(token)); } catch (...) {}
        }
        return output;
    }

    std::string complete(const std::string& prompt) {
        httplib::Client client(server_host_, server_port_);
        client.set_connection_timeout(3, 0);
        client.set_read_timeout(180, 0);

        std::string payload = "{\"model\":\"" + generation_model + "\",\"prompt\":\"" + escapeJsonString(prompt) + "\",\"stream\":false}";
        auto response = client.Post("/api/generate", payload, "application/json");
        if (!response || response->status != 200) {
            return "ERROR: Ollama server unreachable. Ensure 'ollama serve' is running.";
        }

        size_t p = response->body.find("\"response\"");
        if (p == std::string::npos) return "";
        p = response->body.find(':', p) + 1;
        while (p < response->body.size() && (response->body[p] == ' ' || response->body[p] == '\t')) p++;
        if (p >= response->body.size() || response->body[p] != '"') return "";
        p++;

        std::string text_buffer;
        while (p < response->body.size() && response->body[p] != '"') {
            if (response->body[p] == '\\' && p + 1 < response->body.size()) {
                p++;
                if (response->body[p] == 'n') text_buffer += '\n';
                else text_buffer += response->body[p];
            } else {
                text_buffer += response->body[p];
            }
            p++;
        }
        return text_buffer;
    }
};