
# tools/planner/apis.py

import sys
import os
import time

# Let us import project modules like agents.prompts
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from langchain.prompts import PromptTemplate
from agents.prompts import planner_agent_prompt_direct_og

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import tiktoken
import openai

from enum import Enum
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: login to Hugging Face using env token (no hardcoding)
try:
    from huggingface_hub import login
    _hf_env_token = os.environ.get("HF_TOKEN")
    if _hf_env_token:
        try:
            login(token=_hf_env_token)
        except Exception:
            pass
except Exception:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.error.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.error.RateLimitError:
        print("RateLimitError"); time.sleep(60)
    elif error == openai.error.APIError:
        print("APIError")
    elif error == openai.error.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)


class ReflexionStrategy(Enum):
    REFLEXION = 'reflexion'


class Planner:
    """
    Planner that can run local HF models (qwen/phi4/mistral/deepseek/llama/gemma) or hosted models via ChatOpenAI.

    Key behaviors for local HF models:
      - Uses HF token from env (HF_TOKEN) for licensed/gated repos.
      - Prefers safetensors to avoid torch<2.6 .bin restrictions.
      - Ensures PAD token is set (PAD = EOS if missing).
      - Passes EOS/PAD to .generate() for cleaner stops.
    """
    def __init__(
        self,
        agent_prompt: PromptTemplate = planner_agent_prompt_direct_og,
        model_name: str = 'llama'   # default to Llama family
    ) -> None:
        self.agent_prompt = agent_prompt
        self.scratchpad: str = ''
        self.model_name = model_name

        try:
            self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            self.enc = tiktoken.get_encoding("cl100k_base")

        hf_token = os.environ.get("HF_TOKEN")

        # Local HF model path
        if model_name in ['qwen', 'phi4', 'mistral', 'deepseek', 'llama', 'gemma']:
            # Candidate repos (safetensors-first where possible)
            CANDIDATES = {
                'qwen': [
                    "Qwen/Qwen2.5-7B-Instruct",
                ],
                'phi4': [
                    "microsoft/Phi-4-mini-instruct",
                ],
                'mistral': [
                    # Ungated Mistral-architecture instruct models as fallbacks
                    "teknium/OpenHermes-2.5-Mistral-7B",
                    "NousResearch/Hermes-2-Mistral-7B",
                    "OpenOrca/Mistral-7B-OpenOrca",
                    # If you later get access, these will work (requires HF_TOKEN + license accept)
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "mistralai/Mistral-7B-Instruct-v0.3",
                ],
                'deepseek': [
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  # safetensors ✅
                    "deepseek-ai/deepseek-llm-7b-chat",         # may be .bin-only
                ],
                'llama': [
                    "meta-llama/Llama-3.1-8B-Instruct",         # accept license on HF (uses safetensors)
                ],
                'gemma': [
                    "google/gemma-2-9b-it",                     # accept license on HF
                ],
            }

            last_exc: Optional[Exception] = None
            loaded = False
            for candidate in CANDIDATES[model_name]:
                try:
                    # ---- Tokenizer ---------------------------------------------------
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        candidate,
                        use_fast=True,
                        token=hf_token,           # new API
                        use_auth_token=hf_token,  # backwards-compat
                    )
                    if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token

                    # ---- Model -------------------------------------------------------
                    self.model = AutoModelForCausalLM.from_pretrained(
                        candidate,
                        use_safetensors=True,     # prefer safetensors
                        low_cpu_mem_usage=True,
                        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float16),
                        device_map="auto",
                        offload_folder="offload",
                        token=hf_token,
                        use_auth_token=hf_token,
                    )

                    self.enc = self.tokenizer
                    print(f"[Planner] Loaded HF model: {candidate}")
                    loaded = True
                    break

                except Exception as e:
                    last_exc = e
                    print(f"[Planner] Failed to load {candidate}: {e}")

            if not loaded:
                raise RuntimeError(f"Could not load any candidate for {model_name}") from last_exc

        else:
            # Hosted LLM path (OpenAI, etc.)
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                max_tokens=4096,
                openai_api_key=OPENAI_API_KEY
            )

        print(f"PlannerAgent {model_name} loaded.")

    def run(self, text, query, reference_info1,log_file=None) -> str:
        if log_file:
            log_file.write(
                '\n---------------Planner\n' +
                self._build_agent_prompt(text, query, reference_info1)
            )

        prompt = self._build_agent_prompt(text, query, reference_info1)

        # Local HF models
        if self.model_name in ['qwen', 'phi4', 'mistral', 'deepseek', 'llama', 'gemma']:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            output = self.model.generate(
                **inputs,
                max_new_tokens=3072,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                # You can enable sampling if you want more diverse plans:
                # do_sample=True, temperature=0.7, top_p=0.9,
            )

            # Remove echoed prompt if present
            full = self.tokenizer.decode(output[0], skip_special_tokens=True)
            pos = full.find(prompt)
            if pos != -1:
                return full[pos + len(prompt):].strip()
            return full.strip()

        # Hosted LLMs
        input_len = len(self.enc.encode(prompt))
        if input_len > 12000:
            return 'Max Token Length Exceeded.'

        if self.model_name == 'gpt-4o':
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=4096,
                    api_key=OPENAI_API_KEY
                )
                return response['choices'][0]['message']['content']
            except Exception:
                catch_openai_api_error()
                raise

        return self.llm([HumanMessage(content=prompt)]).content

    def _build_agent_prompt(self, text, query, reference_info1) -> str:
        return self.agent_prompt.format(
            text=text,
            query=query,
            reference_info1=reference_info1

        )
