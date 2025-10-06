import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from pathlib import Path
from typing import Dict, List, Optional

import json_repair  # Robust JSON parser for LLM outputs
import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Configuration
INPUT_DIR = "ja-wikipedia/wikipedia_dataset_small/"  # Adjust path as needed
OUTPUT_CSV = "power_systems_qa_dataset.csv"
MODEL_PATH = "kaitchup/Phi-4-AutoRound-GPTQ-4bit"  # Adjust to HF snapshot path
SYSTEM_PROMPT = "電力システムと電気工学の専門家です。言語モデルのトレーニング用の教育コンテンツを生成してください。"

# --- Simple Chunking Configuration ---
CHUNK_SIZE_TOKENS = 500  # Target chunk size in tokens
CHUNK_OVERLAP_TOKENS = 100  # Overlap between chunks in tokens
MAX_MODEL_TOKENS = 8192  # Total model context length
MAX_EXTRA_PROMPT_TOKENS = (
    1000  # Estimated tokens for system prompt, instructions, title, etc.
)
CONCURRENCY = 20  # Number of concurrent sequences for batching
# Calculate maximum tokens available for the content chunk
MAX_CONTENT_TOKENS = MAX_MODEL_TOKENS - MAX_EXTRA_PROMPT_TOKENS
print(f"Maximum tokens allocated for content chunks: {MAX_CONTENT_TOKENS}")

# Pydantic Models (remain the same)
class QuestionList(BaseModel):
    """First pass: List of potential questions"""

    questions: List[str] = Field(
        description="List of self-contained questions that can be answered independently"
    )


class QuestionAnswer(BaseModel):
    """Second pass: Question and Answer pair"""

    question: str = Field(
        description="A clear, self-contained question about power systems"
    )
    answer: str = Field(description="A complete, independent answer to the question")


def read_markdown_files(input_dir: str) -> List[dict]:
    """Read all markdown files from the directory"""
    articles = []
    md_files = list(Path(input_dir).glob("*.md"))

    for filepath in md_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                # Extract title (first line after # header)
                lines = content.split("\n")
                title = lines[0].replace("#", "").strip() if lines else filepath.stem

                # Get content (everything after title)
                text = "\n".join(lines[1:]).strip()

                articles.append(
                    {"title": title, "content": text, "filename": filepath.name}
                )
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    return articles

def remove_surrogates(text):
    if isinstance(text, str):
        # Replace surrogates with replacement char or remove
        return text.encode('utf-8', errors='replace').decode('utf-8')
    return text

# --- Simple Token-Based Chunking ---
def simple_chunk_text(
    text: str, tokenizer, chunk_size: int, overlap: int, max_content_tokens: int
) -> List[str]:
    """
    Chunks text based on token count with overlap.
    Ensures each chunk is within max_content_tokens.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)

    chunks = []
    start_idx = 0

    while start_idx < total_tokens:
        end_idx = start_idx + chunk_size
        # Ensure the chunk doesn't exceed the maximum allowed tokens
        end_idx = min(end_idx, start_idx + max_content_tokens)

        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        chunks.append(chunk_text)

        # Move start index forward by chunk_size - overlap
        # Ensure we don't get stuck in an infinite loop
        if end_idx > start_idx:
            start_idx = end_idx - overlap
            if overlap <= 0 and start_idx < total_tokens:
                # If overlap is 0 and end_idx didn't advance enough, increment by 1
                start_idx = min(start_idx + 1, total_tokens)
        else:
            # Fallback to prevent infinite loop if indices don't advance
            start_idx += chunk_size

        # Break if we've reached the end
        if start_idx >= total_tokens:
            break

    return chunks


# --------------------------------------------


def strip_code_block(text: str) -> str:
    """Strip markdown code blocks from text"""
    if text.startswith("```json"):
        text = text.split("```json")[1].split("```")[0].strip()
    elif text.startswith("```"):
        text = text.split("```")[1].split("```")[0].strip()
    return text


def batch_generate_chat(
    llm, messages_lists: List[List[Dict]], sampling_params: SamplingParams
) -> List[str]:
    """Batch generate chat responses"""
    outputs = llm.chat(messages_lists, sampling_params)
    return [strip_code_block(output.outputs[0].text.strip()) for output in outputs]


def process_articles(articles: List[dict], llm: LLM, tokenizer) -> pd.DataFrame:
    """Process all articles and generate QA dataset with batching"""
    chunks_info: List[Dict] = []

    for article in tqdm(articles, desc="Articles"):
        title = article["title"]
        content = article["content"]

        # Skip very short articles
        if len(content.split()) < 100:
            continue

        # --- Use the new simple token-based chunking method ---
        chunks = simple_chunk_text(
            content,
            tokenizer,
            CHUNK_SIZE_TOKENS,
            CHUNK_OVERLAP_TOKENS,
            MAX_CONTENT_TOKENS,
        )

        for chunk_idx, chunk in enumerate(chunks):
            # Skip very short chunks after tokenization
            if len(chunk.split()) < 50:
                continue

            # Create user prompt for question generation
            user_prompt = f"""以下の電力システムに関するテキストから、次の条件を満たす質問を特定してください：

1. このテキストの情報「のみ」を使って「完全に」答えられる質問
2. 一般的な電力システムの質問（複雑な数式や方程式は避ける）
3. 外部のコンテキストを必要としない、自己完結型の答えを持つ質問
4. 電力システムの理解に役立つ教育的な質問
5. 日本語で記述された質問
6. 【重要】一般知識（GK）や時事ニュース、歴史的事実ではなく、電力システムの「専門知識」に関する質問
7. エンジニアにとって実用的で技術的な内容（例：動作原理、設計概念、システムの動作、技術的な特性など）
8. Wikipediaの一般的な情報ではなく、電力システムのドメイン知識を問う質問

❌ 避けるべき質問例:
- 「いつ発明されましたか？」「誰が開発しましたか？」
- 「どの国で使われていますか？」
- 歴史的な事実や年号

✅ 良い質問例:
- 「なぜこの方式が使われるのですか？」
- 「どのような原理で動作しますか？」
- 「どのような利点と欠点がありますか？」
- 「どのような場合に使用されますか？」

記事タイトル: {title}

テキスト:
{chunk}

これらの基準を満たす専門的・技術的な質問を3〜5個生成してください。各質問は独立して答えられるものにしてください。"""

            # Optional: Double-check the prompt length before adding
            full_prompt_tokens = tokenizer.encode(
                SYSTEM_PROMPT + user_prompt, add_special_tokens=False
            )
            if len(full_prompt_tokens) > MAX_MODEL_TOKENS:
                continue  # Skip this chunk if the prompt is still too long after trimming

            chunks_info.append(
                {
                    "title": title,
                    "chunk": chunk,
                    "chunk_idx": chunk_idx,
                    "user_prompt": user_prompt,
                }
            )

    # Batch generate questions
    qa_prompt_infos: List[Dict] = []
    sampling_params_questions = SamplingParams(
        temperature=0.7, top_p=0.9, max_tokens=512
    )
    json_instruction_questions = '\n\nOutput only valid JSON object: {"questions": ["q1", "q2", ...]} No other text.'

    for i in tqdm(range(0, len(chunks_info), CONCURRENCY), desc="Batch Questions"):
        batch = chunks_info[i : i + CONCURRENCY]
        messages_list = []
        for info in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": info["user_prompt"] + json_instruction_questions,
                },
            ]
            messages_list.append(messages)

        outputs_text = batch_generate_chat(
            llm, messages_list, sampling_params_questions
        )

        for j, text in enumerate(outputs_text):
            try:
                parsed = json_repair.loads(text)
                
                if not isinstance(parsed, dict):
                    raise ValueError(f"Non-dict output from LLM: {repr(text[:200])}")
                
                raw_questions = parsed.get("questions", [])
                
                if not isinstance(raw_questions, list):
                    raise ValueError(f"'questions' field is not a list: {repr(raw_questions)}")
                
                # Filter and clean
                questions = []
                for q in raw_questions:
                    if isinstance(q, str) and q.strip():
                        questions.append(q.strip())
                
                if not questions:
                    raise ValueError("No valid non-empty questions found")
                
                info = batch[j]
                for q in questions:
                    # ... your existing QA prompt logic ...
                    qa_user_prompt = f"""以下のテキストに基づいて、電力システム教育用の明確な質問と回答のペアを作成してください。

記事タイトル: {info['title']}

テキスト:
{info['chunk']}

答えるべき質問: {q}

要件:
1. 質問は明確で自己完結型であること（日本語）
2. 回答は完全で独立していること（日本語）
3. 回答は上記のテキストの情報「のみ」を使用すること
4. 一般的なレベルに保つ - 複雑な数式は避ける
5. LLMのファインチューニングに役立つ教育的な内容にする
6. 【重要】一般知識ではなく、電力システムの専門的・技術的な知識に焦点を当てる
7. エンジニアにとって実用的な内容（動作原理、設計概念、システムの特性など）

最終的な質問と回答のペアを生成してください。技術的な深さを保ちながら、実務に役立つ内容にしてください。"""

                    # Optional: Double-check the QA prompt length before adding
                    full_qa_prompt_tokens = tokenizer.encode(
                        SYSTEM_PROMPT + qa_user_prompt, add_special_tokens=False
                    )
                    if len(full_qa_prompt_tokens) > MAX_MODEL_TOKENS:
                        continue  # Skip this QA pair if the prompt is too long

                    qa_prompt_infos.append(
                        {
                            "title": info["title"],
                            "chunk_idx": info["chunk_idx"],
                            "question": q,
                            "chunk": info["chunk"],
                            "user_prompt": qa_user_prompt,
                        }
                    )
            except Exception as e:
                print(
                    f"JSON parse error for question batch {i//CONCURRENCY +1}, item {j}: {e}"
                )
                continue

    # Batch generate QA pairs
    qa_data = []
    sampling_params_qa = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024)
    json_instruction_qa = '\n\nOutput only valid JSON object: {"question": "the question", "answer": "the answer"} No other text.'

    for i in tqdm(range(0, len(qa_prompt_infos), CONCURRENCY), desc="Batch QA"):
        batch = qa_prompt_infos[i : i + CONCURRENCY]
        messages_list = []
        for info in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": info["user_prompt"] + json_instruction_qa},
            ]
            messages_list.append(messages)

        outputs_text = batch_generate_chat(llm, messages_list, sampling_params_qa)

        for j, text in enumerate(outputs_text):
            try:
                # Use json_repair instead of json.loads
                parsed = json_repair.loads(text)
                qa_pair = QuestionAnswer.model_validate(parsed)
                info = batch[j]
                qa_data.append(
                    {
                        "source_article": info["title"],
                        "chunk_index": info["chunk_idx"],
                        "question": qa_pair.question,
                        "answer": qa_pair.answer,
                        "context": info["chunk"][:200]
                        + "...",  # Store snippet for reference
                    }
                )
            except Exception as e:
                print(
                    f"JSON parse error for QA batch {i//CONCURRENCY +1}, item {j}: {e}"
                )
                continue

    # Create DataFrame
    df = pd.DataFrame(qa_data)
    return df


def main():
    """Main execution function"""
    print("=" * 70)
    print("Power Systems QA Dataset Generator (vLLM Batching)")
    print("=" * 70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Initialize vLLM
    try:
        llm = LLM(
            model=MODEL_PATH,
            max_model_len=MAX_MODEL_TOKENS,  # Explicitly set max_model_len
            max_num_seqs=CONCURRENCY,
            quantization="gptq",
            trust_remote_code=True,
        )
        print(f"✓ vLLM initialized with model: {MODEL_PATH}")
    except Exception as e:
        print(f"✗ vLLM error: {e}")
        return

    # Read articles
    print(f"\nReading articles from: {INPUT_DIR}")
    articles = read_markdown_files(INPUT_DIR)

    if not articles:
        print("No articles found!")
        return

    print(f"✓ Loaded {len(articles)} articles")

    # Process articles and generate QA pairs
    df = process_articles(articles, llm, tokenizer)

    # Clean surrogate characters in all string columns
    text_columns = ['source_article', 'question', 'answer', 'context']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(remove_surrogates)

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # Print summary
    print("\n" + "=" * 70)
    print("Dataset Generation Complete!")
    print("=" * 70)
    print(f"Total QA pairs: {len(df)}")
    print(f"Unique articles: {df['source_article'].nunique()}")
    print(f"Output saved to: {OUTPUT_CSV}")
    print("\nSample QA pairs:")
    print(df[["question", "answer"]].head(3).to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
