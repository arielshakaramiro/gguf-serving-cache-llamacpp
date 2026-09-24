# GGUF Serving + SQL Response Cache (llama.cpp)

Serves a fine-tuned GGUF model with [llama.cpp](https://github.com/ggml-org/llama.cpp), with a response cache in front of it, and a deliberate demonstration of the main risk that comes with caching LLM output: a wrong answer, once cached, stays wrong until something other than the cache changes.

This notebook is meant to be paired with **[llama3-qlora-finetuning](https://github.com/arielshakaramiro/llama3-qlora-finetuning)** — it loads the GGUF file that repo exports. If that file isn't found in Google Drive, it falls back to a public model on Hugging Face instead.

Everything below comes from an actual Google Colab run (GPU: NVIDIA A100-SXM4-40GB). No number is estimated.

## What this does

1. Installs `llama-cpp-python` compiled with CUDA support for whatever GPU is detected.
2. Loads the GGUF model — from the companion repo's export on Drive, or a Hugging Face fallback.
3. Benchmarks the same prompt on CPU-only vs. full GPU offload.
4. Builds a cache layer keyed by a SHA-256 hash of the instruction, input, model tag, and decoding parameters — so changing any of those automatically invalidates old entries. Backends: SQLite (default) or MySQL (credentials via Colab Secrets, never hardcoded).
5. Measures cache-miss vs. cache-hit latency on the same prompts.
6. Runs a small ordinal-reasoning test ("who is the Nth president of Konoha," given a list) with a programmatically checkable answer key, then demonstrates what happens when a wrong answer gets cached.

## Results

### GPU offload

| | Tokens/sec |
|---|---|
| CPU (`n_gpu_layers=0`) | 5.51 |
| GPU (`n_gpu_layers=-1`) | 115.47 |
| **Speedup** | **21.0x** |

![CPU vs GPU throughput](images/cpu_vs_gpu.png)

### Cache latency

| Prompt | Cache miss (model) | Cache hit (SQLite) |
|---|---|---|
| "siapa presiden pertama di indonesia?" | 250 ms | 0.06 ms |
| "ibu kota indonesia adalah" | 80 ms | 0.06 ms |
| "Apa itu algoritma pemrograman?" | 1,450 ms | 0.06 ms |

![Cache miss vs hit latency](images/cache_latency.png)

The gap is 3–4 orders of magnitude, which is expected: a cache hit is a SQLite primary-key lookup, a cache miss is a full generation pass.

### The risk the cache doesn't protect against

The ordinal-reasoning test scored **60% (3/5)**. Both wrong answers followed the same pattern — the model repeated the *previous* correct name instead of advancing to the next one in the list (asked for the 3rd president, it answered with the 2nd's name; asked for the 5th, it answered with the 4th's).

Using one of those wrong answers, the notebook then shows the actual risk: the model is called once (wrong answer), the same wrong answer is served from cache on a second call, the cache entry is deleted, and a third call **produces the identical wrong answer again** — because decoding is greedy and deterministic. Deleting a bad cache entry doesn't fix a systematically wrong answer; only changing the prompt, the data, or the model does.

## How to run

1. Open `notebooks/llama_cpp_gguf_serving_cache.ipynb` in Google Colab.
2. Set the runtime to a GPU.
3. Run all cells top to bottom. `llama-cpp-python` is compiled from source with CUDA flags matching the detected GPU — this step can take several minutes.
4. To use MySQL instead of the default SQLite cache, set `USE_MYSQL = True` and provide `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DB` in Colab Secrets (or enter them manually when prompted).

## Repo structure

```
.
├── notebooks/
│   └── llama_cpp_gguf_serving_cache.ipynb
├── images/
│   ├── cpu_vs_gpu.png
│   └── cache_latency.png
├── LICENSE
└── README.md
```

## Limitations

- The cache is exact-match: two questions with the same meaning but different wording are treated as unrelated and both hit the model. A semantic cache would need embeddings and a similarity threshold.
- The cache has no way to judge correctness — a wrong answer that finishes generating normally is cached just like a right one.
- Caching only makes sense with deterministic (greedy) decoding. With sampling enabled, identical prompts can legitimately produce different answers, and the cache would freeze one arbitrary variant.
- The benchmark numbers are single measurements on one Colab session; GPU throughput in particular can vary run to run.

## License

MIT — see [LICENSE](LICENSE).
