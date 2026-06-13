#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""apply_eval_callback_patch.py — patch llama.cpp's eval-callback to dump binary tensors.

Phase L.1.0a fixture-capture mechanism: when LLAMA_DUMP_DIR env var is set,
each operation in the ggml graph emits a .bin file containing dtype + dims +
strides + raw data. Plus a manifest.tsv listing them all in order.

File layout per .bin:
  uint32_t dtype_code   (0=F32, 1=F16, 2=I32, 3=I16, 4=I8, 5=Quantized-raw)
  uint32_t n_dims       (always 4)
  int64_t  ne[4]        dimensions
  uint64_t nb[4]        strides (bytes)
  uint64_t n_bytes      payload size
  uint8_t  data[n_bytes]
"""
import sys

SRC_PATH = "/tmp/llama_cpp_test/examples/eval-callback/eval-callback.cpp"

with open(SRC_PATH) as f:
    content = f.read()

# 1. Add includes
old_includes = "#include <cstdio>\n#include <string>\n#include <vector>\n"
new_includes = (
    "#include <cstdio>\n"
    "#include <string>\n"
    "#include <vector>\n"
    "#include <cstdlib>\n"
    "#include <cstring>\n"
    "#include <sys/stat.h>\n"
    "#include <atomic>\n"
)
if old_includes in content:
    content = content.replace(old_includes, new_includes)
else:
    sys.exit("ERROR: includes block not found; cannot patch")

# 2. Insert helper before ggml_debug
helper = r"""
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

    char manifest_path[512];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.tsv", dump_dir);
    FILE * mfp = std::fopen(manifest_path, "a");
    if (mfp) {
        std::fprintf(mfp, "%04d\t%s\t%s\t%s\t%lld,%lld,%lld,%lld\n",
                     idx, t->name, ggml_op_desc(t), ggml_type_name(t->type),
                     (long long)t->ne[0], (long long)t->ne[1],
                     (long long)t->ne[2], (long long)t->ne[3]);
        std::fclose(mfp);
    }
}

"""

old_decl = "static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {"
if old_decl in content:
    content = content.replace(old_decl, helper + old_decl)
else:
    sys.exit("ERROR: ggml_debug declaration not found; cannot patch")

# 3. Insert dump call inside ggml_debug
old_block = (
    "    if (!ggml_is_quantized(t->type)) {\n"
    "        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();\n"
    "        ggml_print_tensor(data, t->type, t->ne, t->nb, 3);\n"
    "    }"
)
new_block = (
    "    {\n"
    "        // Materialize raw bytes for both unquantized and quantized tensors\n"
    "        uint8_t * raw = is_host ? (uint8_t *) t->data : cb_data->data.data();\n"
    "        if (!ggml_is_quantized(t->type)) {\n"
    "            ggml_print_tensor(raw, t->type, t->ne, t->nb, 3);\n"
    "        }\n"
    "        dump_tensor_binary(t, raw);\n"
    "\n"
    "        // Also dump source leaf tensors (inputs that are constants, not graph nodes).\n"
    "        // These animate as inputs to ops but aren't themselves emitted by the callback.\n"
    "        for (int src_i = 0; src_i < GGML_MAX_SRC; ++src_i) {\n"
    "            const struct ggml_tensor * src = t->src[src_i];\n"
    "            if (!src) continue;\n"
    "            // Only dump small/leaf-like inputs (not gigantic weight tensors).\n"
    "            // Heuristic: dump if total bytes < 1 MB (covers token IDs, position arrays,\n"
    "            // small embedding tables for layer-0 fixtures).\n"
    "            if (ggml_nbytes(src) >= (1 << 20)) continue;\n"
    "            // Avoid re-dumping the same tensor multiple times (heuristic by ptr).\n"
    "            static std::vector<const void *> seen_srcs;\n"
    "            bool already = false;\n"
    "            for (const void * p : seen_srcs) if (p == src) { already = true; break; }\n"
    "            if (already) continue;\n"
    "            seen_srcs.push_back(src);\n"
    "            // Materialize src bytes.\n"
    "            std::vector<uint8_t> src_buf;\n"
    "            const bool src_host = ggml_backend_buffer_is_host(src->buffer);\n"
    "            uint8_t * src_raw;\n"
    "            if (src_host) {\n"
    "                src_raw = (uint8_t *) src->data;\n"
    "            } else {\n"
    "                src_buf.resize(ggml_nbytes(src));\n"
    "                ggml_backend_tensor_get(src, src_buf.data(), 0, ggml_nbytes(src));\n"
    "                src_raw = src_buf.data();\n"
    "            }\n"
    "            dump_tensor_binary((struct ggml_tensor *) src, src_raw);\n"
    "        }\n"
    "    }"
)
if old_block in content:
    content = content.replace(old_block, new_block)
else:
    sys.exit("ERROR: print_tensor block not found; cannot patch")

with open(SRC_PATH, "w") as f:
    f.write(content)
print(f"Patched {SRC_PATH}")
print(f"New file size: {len(content)} bytes")
