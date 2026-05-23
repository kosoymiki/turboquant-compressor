/**
 * TurboQuant Core — CLI entry point
 *
 * Usage:
 *   turboquant_core compress [--dims N] [--bits N] [--seed N] [--qjl-dims N] [--codebook-type uniform|beta] < vectors.json
 *   turboquant_core search   [--k N] [--metric cosine|dot] [--no-qjl] < query.json
 *   turboquant_core header   < database.b64
 *   turboquant_core quality  [--iterations N] [--codebook-type uniform|beta] < vectors.json
 *   turboquant_core version
 *
 * Input format (JSON):
 *   compress: { "vectors": [[...], ...], "dims": N, "bits": 4, "seed": 42, "qjl_dims": 16 }
 *   search:   { "db": "...base64...", "query": [...], "dims": N, "k": 5, "metric": "cosine", "use_qjl": 1 }
 *   quality:  { "vectors": [[...], ...], "dims": N, "bits": 4, "seed": 42, "qjl_dims": 16, "iterations": N }
 *
 * Output: JSON with stats + result
 */

#include "tq_core.h"
#include "tq_quality.h"
#include "tq_limits.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>

static std::string read_stdin_all() {
    std::string s;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), stdin)) > 0) {
        s.append(buf, n);
    }
    return s;
}

static void trim(std::string& s) {
    size_t a = 0;
    while (a < s.size() && (s[a] == ' ' || s[a] == '\n' || s[a] == '\r' || s[a] == '\t')) a++;
    size_t b = s.size();
    while (b > a && (s[b-1] == ' ' || s[b-1] == '\n' || s[b-1] == '\r' || s[b-1] == '\t')) b--;
    s = s.substr(a, b - a);
}

static int cmd_version() {
    printf("turboquant_core %s\n", TQ_CORE_VERSION);
    printf("  TQMC format v3, 1-bit Hadamard QJL v2\n");
    return 0;
}

static int cmd_header(const char* b64) {
    tq_db_header h;
    int r = tq_decode_header(b64, &h);
    if (r != TQ_OK) {
        fprintf(stderr, "tq_decode_header failed: %d\n", r);
        return 1;
    }
    printf("{\n");
    printf("  \"magic\": 0x%08X,\n", h.magic);
    printf("  \"version\": %u,\n", h.version);
    printf("  \"dimensions\": %u,\n", h.dimensions);
    printf("  \"padded_dimensions\": %u,\n", h.padded_dimensions);
    printf("  \"vector_count\": %u,\n", h.vector_count);
    printf("  \"bits_per_value\": %u,\n", h.bits_per_value);
    printf("  \"rotation_seed\": %u,\n", h.rotation_seed);
    printf("  \"qjl_dims\": %u,\n", h.qjl_dims);
    printf("  \"qjl_bytes\": %u,\n", h.qjl_bytes);
    printf("  \"has_qjl\": %s\n", h.has_qjl ? "true" : "false");
    printf("}\n");
    return 0;
}

static int cmd_compress(int argc, char** argv) {
    uint32_t dims = 1024, seed = 42, N = 0;
    uint8_t bits = 4, qjl_dims = 16;
    uint8_t codebook_type = TQ_CODEBOOK_UNIFORM;
    bool use_qjl = true;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--dims") == 0 && i + 1 < argc) dims = atoi(argv[++i]);
        else if (strcmp(argv[i], "--bits") == 0 && i + 1 < argc) bits = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--qjl-dims") == 0 && i + 1 < argc) qjl_dims = atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-qjl") == 0) use_qjl = false;
        else if (strcmp(argv[i], "--codebook-type") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "beta") == 0) codebook_type = TQ_CODEBOOK_BETA;
        }
    }

    std::string input = read_stdin_all();
    trim(input);

    // Parse JSON: { "vectors": [[...], ...] }
    // Simple parsing: find first "[[" and last "]]"
    size_t start = input.find("[[");
    size_t end = input.rfind("]]");
    if (start == std::string::npos || end == std::string::npos) {
        fprintf(stderr, "Input must contain [[...]] array of vectors\n");
        return 1;
    }

    // Count vectors: count "[[" occurrences
    size_t count_start = 0;
    N = 0;
    size_t pos = start;
    while ((pos = input.find("[[", pos)) != std::string::npos && pos < end) {
        N++; pos++;
    }
    if (N == 0) { fprintf(stderr, "No vectors found\n"); return 1; }

    std::vector<float> vectors((size_t)N * dims);
    // Extract numbers with a simple scanner
    size_t num_start = start;
    size_t vidx = 0;
    for (size_t i = start; i < end && vidx < vectors.size(); i++) {
        char c = input[i];
        if ((c >= '0' && c <= '9') || c == '-' || c == '.') {
            // parse number
            size_t j = i;
            if (input[j] == '-') j++;
            while (j < input.size() && ((input[j] >= '0' && input[j] <= '9') || input[j] == '.')) j++;
            std::string num = input.substr(i, j - i);
            vectors[vidx++] = (float)atof(num.c_str());
            i = j - 1;
        }
    }

    if (vidx != vectors.size()) {
        fprintf(stderr, "Parsed %zu values, expected %zu\n", vidx, (size_t)N * dims);
        return 1;
    }

    uint8_t qjl = use_qjl ? qjl_dims : 0;
    size_t out_len = tq_estimate_compressed_size(N, dims, bits, qjl) * 2; // generous

    std::string out(out_len + 1, '\0');
    tq_compress_stats st;
    int r = tq_compress(vectors.data(), N, dims, bits, seed, qjl, codebook_type,
                        out.data(), &out_len, &st);
    if (r != TQ_OK) {
        fprintf(stderr, "tq_compress failed: %d\n", r);
        return 1;
    }
    out.resize(out_len);

    printf("{\n");
    printf("  \"ok\": true,\n");
    printf("  \"compression_ratio\": %.4f,\n", st.compression_ratio);
    printf("  \"compressed_bytes\": %u,\n", st.compressed_bytes);
    printf("  \"original_bytes\": %u,\n", st.original_bytes);
    printf("  \"vector_count\": %u,\n", st.vector_count);
    printf("  \"dimensions\": %u,\n", st.dimensions);
    printf("  \"qjl_dims\": %u,\n", st.qjl_dims);
    printf("  \"qjl_bytes\": %u,\n", st.qjl_bytes);
    printf("  \"has_qjl\": %s,\n", st.has_qjl ? "true" : "false");
    printf("  \"compress_ns\": %u,\n", st.compress_ns);
    printf("  \"database\": \"%s\"\n", out.c_str());
    printf("}\n");

    return 0;
}

static int cmd_search(int argc, char** argv) {
    uint32_t k = 5, dims = 1024;
    uint8_t metric = 0; // 0=cosine, 1=dot
    uint8_t use_qjl = 1;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dims") == 0 && i + 1 < argc) dims = atoi(argv[++i]);
        else if (strcmp(argv[i], "--metric") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "dot") == 0) metric = 1;
        } else if (strcmp(argv[i], "--no-qjl") == 0) use_qjl = 0;
    }

    std::string input = read_stdin_all();
    trim(input);

    // Parse: { "db": "...", "query": [...] }
    size_t db_start = input.find("\"db\"");
    size_t q_start = input.find("\"query\"");
    if (db_start == std::string::npos || q_start == std::string::npos) {
        fprintf(stderr, "Input must contain { \"db\": \"...\", \"query\": [...] }\n");
        return 1;
    }

    // Extract base64 between first two quotes after "db"
    size_t q1 = input.find('"', input.find("\"db\"") + 4);
    size_t q2 = input.find('"', q1 + 1);
    std::string db_b64 = input.substr(q1 + 1, q2 - q1 - 1);

    // Extract query floats
    size_t qs1 = input.find('[', q_start);
    size_t qs2 = input.find(']', qs1);
    std::string qnums = input.substr(qs1 + 1, qs2 - qs1 - 1);

    std::vector<float> query(dims);
    size_t pos = 0;
    uint32_t qi = 0;
    while (pos < qnums.size() && qi < dims) {
        while (pos < qnums.size() && qnums[pos] == ' ') pos++;
        if (pos >= qnums.size()) break;
        size_t j = pos;
        if (qnums[j] == '-' || (qnums[j] >= '0' && qnums[j] <= '9')) {
            if (qnums[j] == '-') j++;
            while (j < qnums.size() && ((qnums[j] >= '0' && qnums[j] <= '9') || qnums[j] == '.')) j++;
            query[qi++] = (float)atof(qnums.substr(pos, j - pos).c_str());
            pos = j;
        } else {
            pos++;
        }
    }

    std::vector<uint32_t> indices(k);
    std::vector<float> scores(k);
    tq_search_stats st;
    int r = tq_search(db_b64.c_str(), query.data(), dims, k, metric, use_qjl,
                      indices.data(), scores.data(), &st);

    if (r != TQ_OK) {
        fprintf(stderr, "tq_search failed: %d\n", r);
        return 1;
    }

    printf("{\n");
    printf("  \"ok\": true,\n");
    printf("  \"vector_count\": %u,\n", st.vector_count);
    printf("  \"searched_count\": %u,\n", st.searched_count);
    printf("  \"qjl_applied\": %s,\n", st.qjl_applied ? "true" : "false");
    printf("  \"search_ns\": %u,\n", st.search_ns);
    printf("  \"results\": [");
    for (uint32_t i = 0; i < k; i++) {
        if (i > 0) printf(", ");
        printf("{\"index\": %u, \"score\": %.6f}", indices[i], scores[i]);
    }
    printf("]\n");
    printf("}\n");

    return 0;
}

static int cmd_quality(int argc, char** argv) {
    uint32_t dims = 1024, iterations = 0;
    uint8_t codebook_type = TQ_CODEBOOK_UNIFORM;

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--dims") == 0 && i + 1 < argc) dims = atoi(argv[++i]);
        else if (strcmp(argv[i], "--codebook-type") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "beta") == 0) codebook_type = TQ_CODEBOOK_BETA;
        }
        else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
    }

    std::string input = read_stdin_all();
    trim(input);

    size_t start = input.find("[[");
    size_t end = input.rfind("]]");
    if (start == std::string::npos || end == std::string::npos) {
        fprintf(stderr, "Input must contain [[...]] array of vectors\n");
        return 1;
    }

    uint32_t N = 0;
    size_t pos = start;
    while ((pos = input.find("[[", pos)) != std::string::npos && pos < end) { N++; pos++; }
    if (N == 0) { fprintf(stderr, "No vectors found\n"); return 1; }

    std::vector<float> vectors((size_t)N * dims);
    size_t vidx = 0;
    for (size_t i = start; i < end && vidx < vectors.size(); i++) {
        char c = input[i];
        if ((c >= '0' && c <= '9') || c == '-' || c == '.') {
            size_t j = i;
            if (input[j] == '-') j++;
            while (j < input.size() && ((input[j] >= '0' && input[j] <= '9') || input[j] == '.')) j++;
            vectors[vidx++] = (float)atof(input.substr(i, j - i).c_str());
            i = j - 1;
        }
    }

    if (vidx != vectors.size()) {
        fprintf(stderr, "Parsed %zu values, expected %zu\n", vidx, (size_t)N * dims);
        return 1;
    }

    // Compress first
    size_t est = tq_estimate_compressed_size(N, dims, 4, 0);
    std::string db(est * 2, '\0');
    size_t db_len = est * 2;
    tq_compress_stats cst;
    int r = tq_compress(vectors.data(), N, dims, 4, 42, 0, codebook_type,
                        db.data(), &db_len, &cst);
    if (r != TQ_OK) { fprintf(stderr, "compress failed: %d\n", r); return 1; }
    db.resize(db_len);

    // Run quality benchmark
    tq_quality_config cfg = {5, iterations, 0, codebook_type};
    tq_quality_result qres = tq_quality_benchmark(
        vectors.data(), N, dims, db.c_str(), cfg);

    printf("{\n");
    printf("  \"compression_ratio\": %.4f,\n", cst.compression_ratio);
    printf("  \"mse_per_coord\": %.6f,\n", qres.mse_per_coord);
    printf("  \"recall_at_1\": %.4f,\n", qres.recall_at_1);
    printf("  \"recall_at_5\": %.4f,\n", qres.recall_at_5);
    printf("  \"avg_search_ns\": %.2f,\n", qres.avg_search_ns);
    printf("  \"codebook_type\": \"%s\",\n",
           codebook_type == TQ_CODEBOOK_BETA ? "beta" : "uniform");
    printf("  \"iterations\": %u\n", qres.total_queries);
    printf("}\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: turboquant_core <compress|search|header|version> [args]\n");
        return 1;
    }

    if (strcmp(argv[1], "version") == 0) return cmd_version();

    if (strcmp(argv[1], "header") == 0) {
        std::string b64 = read_stdin_all();
        trim(b64);
        return cmd_header(b64.c_str());
    }

    if (strcmp(argv[1], "compress") == 0) {
        return cmd_compress(argc - 2, argv + 2);
    }
    if (strcmp(argv[1], "search") == 0) {
        return cmd_search(argc - 2, argv + 2);
    }

    if (strcmp(argv[1], "quality") == 0) {
        std::string b64 = read_stdin_all();
        trim(b64);
        // Parse quality input from stdin
        return cmd_quality(argc - 2, argv + 2);
    }

    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    return 1;
}