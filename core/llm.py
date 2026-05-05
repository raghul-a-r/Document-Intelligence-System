from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from llama_cpp import Llama

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)


llm = Llama(
    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_gpu_layers=35,
    n_ctx=4096,
    n_threads=8
)


def generate_answer(query, context):
    prompt = f"""[INST]
You are an expert AI assistant.

Answer clearly using the context below.

Context:
{context}

Question:
{query}

Answer in 3-5 sentences.
[/INST]
"""

    output = llm(prompt, max_tokens=200, temperature=0.2)

    return output["choices"][0]["text"].strip()
