ENDPOINTS = {
    "kaggle_2xT4_tp2": "https://rufflike-heteromorphic-lacie.ngrok-free.dev",
    "colab_A100_fp16": "",
    # add colab_A100_int8 once you run the quantized variant
}

MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Sweep parameters — these define your experiment matrix
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
PROMPT_CONFIGS = {
    "short_64":   {"tokens": 64,   "text": "Explain what attention is in one sentence."},
    "medium_256": {"tokens": 256,  "text": "Explain the history of deep learning in detail. " * 6},
    "long_1024":  {"tokens": 1024, "text": "Write a comprehensive explanation of how transformers work. " * 18},
}
OUTPUT_TOKENS = 128   # fixed output length for fair TPOT comparison
WARMUP_REQUESTS = 3   # discard these — CUDA kernel cold-start
REQUESTS_PER_CELL = 20