import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import multiprocessing as mp
import sys
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from text import WIKI_INPUT

# ----------------------------
# CONFIGURATION
# ----------------------------
MODEL_NAME = "kaitchup/Phi-4-AutoRound-GPTQ-4bit"
MAX_MODEL_LEN = 4192  # total context length
INPUT_TOKENS = 3000
OUTPUT_TOKENS = 1000
BATCH_SIZE = 10  # <-- Change this to your desired batch size

if __name__ == "__main__":
    mp.set_start_method("spawn")
    # ----------------------------
    # SETUP
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.95,  # Use more VRAM for larger batches
        trust_remote_code=True,
        enforce_eager=False,  # enables cudagraphs for speed
        max_num_seqs=20,
    )

    # Trim input to 3000 tokens
    def trim_to_tokens(text: str, max_tokens: int) -> str:
        tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return tokenizer.decode(tokens, skip_special_tokens=True)

    trimmed_input = trim_to_tokens(WIKI_INPUT, INPUT_TOKENS)

    # Create batch
    prompts = [trimmed_input] * BATCH_SIZE
    messages_list = [
        [
            {
                "role": "user",
                "content": f"Summarize the following text in more than {OUTPUT_TOKENS} word, write an essay:\n\n{p}",
            }
        ]
        for p in prompts
    ]

    sampling_params = SamplingParams(
        temperature=0.0,  # deterministic output for evaluation
        top_p=0.1,
        max_tokens=OUTPUT_TOKENS,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # ----------------------------
    # INFERENCE
    # ----------------------------
    print(f"Running batch inference (batch_size={BATCH_SIZE})...")
    start_time = time.time()
    outputs = llm.chat(messages_list, sampling_params)
    end_time = time.time()

    total_time = end_time - start_time
    throughput = BATCH_SIZE / total_time

    # Token counting
    total_input_tokens = 0
    total_output_tokens = 0
    for i, output in enumerate(outputs):
        inp_toks = len(tokenizer(prompts[i]).input_ids)
        out_toks = len(tokenizer(output.outputs[0].text).input_ids)
        total_input_tokens += inp_toks
        total_output_tokens += out_toks
        print(f"\n--- Summary {i+1} ---")
        if i == 0:  # print only the first summary to avoid clutter
            print(output.outputs[0].text)

    total_tokens = total_input_tokens + total_output_tokens
    tokens_per_sec = total_tokens / total_time

    # ----------------------------
    # REPORT
    # ----------------------------
    print("\n" + "=" * 60)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total time: {total_time:.2f} sec")
    print(f"Throughput: {throughput:.2f} requests/sec")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Tokens/sec: {tokens_per_sec:.2f}")
    print("=" * 60)
