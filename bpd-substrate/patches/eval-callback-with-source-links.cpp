#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <atomic>
#include <unordered_map>
static std::unordered_map<const void*, int> g_tensor_to_idx;

/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
 */
struct callback_data {
    std::vector<uint8_t> data;
};

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                LOG("                                      ..., \n");
                i2 = ne[2] - n;
            }
            LOG("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    LOG("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                LOG("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        LOG("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        GGML_ABORT("fatal error");
                    }
                    LOG("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) LOG(", ");
                }
                LOG("],\n");
            }
            LOG("                                      ],\n");
        }
        LOG("                                     ]\n");
        LOG("                                     sum = %f\n", sum);
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */

// Phase L.1.0a: optional binary tensor dump for our kernels vs llama.cpp gates
static std::atomic<int> g_dump_step{0};
static void dump_tensor_binary(struct ggml_tensor * t, uint8_t * data) {
    const char * dump_dir = std::getenv("LLAMA_DUMP_DIR");
    if (!dump_dir || !*dump_dir) return;
    static bool dir_made = false;
    if (!dir_made) { mkdir(dump_dir, 0755); dir_made = true; }

    int idx = g_dump_step.fetch_add(1);
    // Sanitize tensor name
    char safe_name[128];
    snprintf(safe_name, sizeof(safe_name), "%s", t->name);
    for (char * p = safe_name; *p; ++p) {
        char c = *p;
        if (c == '/' || c == ' ' || c == '(' || c == ')') *p = '_';
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/%04d_%s.bin", dump_dir, idx, safe_name);
    FILE * fp = std::fopen(path, "wb");
    if (!fp) { std::fprintf(stderr, "dump_tensor_binary: cannot open %s\n", path); return; }

    uint32_t dtype_code = 0;
    switch (t->type) {
        case GGML_TYPE_F32: dtype_code = 0; break;
        case GGML_TYPE_F16: dtype_code = 1; break;
        case GGML_TYPE_I32: dtype_code = 2; break;
        case GGML_TYPE_I16: dtype_code = 3; break;
        case GGML_TYPE_I8:  dtype_code = 4; break;
        default: dtype_code = 5; break;
    }
    uint32_t n_dims = 4;
    std::fwrite(&dtype_code, sizeof(dtype_code), 1, fp);
    std::fwrite(&n_dims,     sizeof(n_dims),     1, fp);
    std::fwrite(t->ne,       sizeof(int64_t),    4, fp);
    std::fwrite(t->nb,       sizeof(size_t),     4, fp);

    uint64_t n_bytes = ggml_nbytes(t);
    std::fwrite(&n_bytes, sizeof(n_bytes), 1, fp);
    std::fwrite(data,     1, n_bytes, fp);
    std::fclose(fp);
    g_tensor_to_idx[(const void*)t] = idx;

    char manifest_path[512];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.tsv", dump_dir);
    FILE * mfp = std::fopen(manifest_path, "a");
    if (mfp) {
        std::string src_str;
        for (int si = 0; si < GGML_MAX_SRC; si++) {
            if (!t->src[si]) break;
            auto it = g_tensor_to_idx.find((const void*)t->src[si]);
            if (it != g_tensor_to_idx.end()) {
                if (!src_str.empty()) src_str += ",";
                src_str += std::to_string(it->second);
            }
        }
        std::fprintf(mfp, "%04d\t%s\t%s\t%s\t%lld,%lld,%lld,%lld\t%s\n",
                     idx, t->name, ggml_op_desc(t), ggml_type_name(t->type),
                     (long long)t->ne[0], (long long)t->ne[1],
                     (long long)t->ne[2], (long long)t->ne[3],
                     src_str.c_str());
        std::fclose(mfp);
    }
}

static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return true; // Always retrieve data
    }

    char src1_str[128] = {0};
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
         t->name, ggml_type_name(t->type), ggml_op_desc(t),
         src0->name, ggml_ne_string(src0).c_str(),
         src1 ? src1_str : "",
         ggml_ne_string(t).c_str());


    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    {
        // Materialize raw bytes for both unquantized and quantized tensors
        uint8_t * raw = is_host ? (uint8_t *) t->data : cb_data->data.data();
        if (!ggml_is_quantized(t->type)) {
            ggml_print_tensor(raw, t->type, t->ne, t->nb, 3);
        }
        dump_tensor_binary(t, raw);

        // Also dump source leaf tensors (inputs that are constants, not graph nodes).
        // These animate as inputs to ops but aren't themselves emitted by the callback.
        for (int src_i = 0; src_i < GGML_MAX_SRC; ++src_i) {
            const struct ggml_tensor * src = t->src[src_i];
            if (!src) continue;
            // Only dump small/leaf-like inputs (not gigantic weight tensors).
            // Heuristic: dump if total bytes < 1 MB (covers token IDs, position arrays,
            // small embedding tables for layer-0 fixtures).
            if (ggml_nbytes(src) >= (1 << 20)) continue;
            // Avoid re-dumping the same tensor multiple times (heuristic by ptr).
            static std::vector<const void *> seen_srcs;
            bool already = false;
            for (const void * p : seen_srcs) if (p == src) { already = true; break; }
            if (already) continue;
            seen_srcs.push_back(src);
            // Materialize src bytes.
            std::vector<uint8_t> src_buf;
            const bool src_host = ggml_backend_buffer_is_host(src->buffer);
            uint8_t * src_raw;
            if (src_host) {
                src_raw = (uint8_t *) src->data;
            } else {
                src_buf.resize(ggml_nbytes(src));
                ggml_backend_tensor_get(src, src_buf.data(), 0, ggml_nbytes(src));
                src_raw = src_buf.data();
            }
            dump_tensor_binary((struct ggml_tensor *) src, src_raw);
        }
    }

    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    callback_data cb_data;

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ggml_debug;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
