import os
from openai import OpenAI
import anthropic
import google.generativeai as genai  # Add this import
from together import Together
from ollama import chat
from ollama import ChatResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

class LLMProviderInterface:
    """
    A common interface for LLM calls.
    """
    def get_response(self, model: str, prompt: str) -> str:
        raise NotImplementedError("Subclasses should implement this method.")


class OpenAIProvider(LLMProviderInterface):
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def get_response(self, model: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,
        )
        return response.choices[0].message.content.strip()


class AnthropicProvider(LLMProviderInterface):
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def get_response(self, model: str, prompt: str) -> str:
        # According to Anthropic docs, this is one way to call the API.
        response = self.client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    
class GeminiProvider(LLMProviderInterface):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        
    def get_response(self, model: str, prompt: str) -> str:
        model = genai.GenerativeModel(model)
        response = model.generate_content(
            contents=prompt,
            generation_config={
                "max_output_tokens": 4096,
            },
            stream=False
        )
        return response.text.strip()
    

class TogetherProvider(LLMProviderInterface):
    def __init__(self, api_key: str):
        self.client = Together(api_key=api_key)

    def get_response(self, model: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()

class OllamaProvider(LLMProviderInterface):
    def __init__(self, url: str = "http://localhost:11434"):
        self.url = url

    def get_response(self, model: str, prompt: str) -> str:
        model = model[len("ollama-"):] if model.lower().startswith("ollama-") else model
        response: ChatResponse = chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ])
        return response.message.content.strip()

class LocalCheckpointProvider(LLMProviderInterface):
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        # Load the base model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        from pathlib import Path
        import json
        
        # Convert checkpoint_path to Path object
        checkpoint_path = Path(self.checkpoint_path)
        
        # Find config file in parent directory
        config_path = checkpoint_path.parent.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Get base model name from config
        base_model_name = config['model']['name']
        if not base_model_name:
            raise ValueError("Base model name not found in config")
        
        # Load metrics file to get additional info if available
        metrics_file = checkpoint_path / "metrics.json"
        metrics = {}
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
        
        print(f"Loading model from {checkpoint_path}, base model name: {base_model_name}")

        # Load the base model and tokenizer with matching configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            padding_side="left",
            trust_remote_code=True
        )
        
        # Load and apply the LoRA weights
        self.model = PeftModel.from_pretrained(
            self.model,
            os.path.join(self.checkpoint_path, "model"),
            is_trainable=False
        )
        self.model.eval()
    
    def get_response(self, model: str, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            self._load_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        print("Generating response...")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=1.5,
            top_p=0.8,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response generated: {response}")
        return response[len(prompt):].strip()

def create_llm_provider(model: str) -> LLMProviderInterface:
    """
    Factory function for creating an LLM provider instance.
    If any substring in the openai_substrings list is found in the model name (case-insensitive),
    returns an instance of OpenAIProvider.
    If any substring in the anthropic_substrings list is found, returns an instance of AnthropicProvider.
    If the model starts with 'local:', returns a LocalCheckpointProvider.
    Otherwise, raises a ValueError.
    """
    model_lower = model.lower()
    openai_substrings = ["gpt-", "o1-", "o3-"]
    anthropic_substrings = ["claude"]
    gemini_substrings = ["gemini"]
    together_substrings = ["meta-llama", "deepseek", "Gryphe", "microsoft", "mistralai", "NousResearch", "nvidia", "Qwen", "upstage"]
    ollama_substrings = ["ollama-"]

    if model_lower.startswith("local:"):
        checkpoint_path = model[6:]  # Remove "local:" prefix
        # Convert relative path to absolute path based on script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.abspath(os.path.join(script_dir, checkpoint_path))
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        return LocalCheckpointProvider(checkpoint_path=checkpoint_path)
    elif any(substr.lower() in model_lower for substr in openai_substrings):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        return OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
    elif any(substr.lower() in model_lower for substr in anthropic_substrings):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY is not set in the environment variables.")
        return AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif any(substr.lower() in model_lower for substr in gemini_substrings):
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")
        return GeminiProvider(api_key=os.getenv("GOOGLE_API_KEY"))
    elif any(substr.lower() in model_lower for substr in ollama_substrings):
        return OllamaProvider(url=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    elif any(substr.lower() in model_lower for substr in together_substrings):
        if not os.getenv("TOGETHERAI_API_KEY"):
            raise ValueError("TOGETHERAI_API_KEY is not set in the environment variables.")
        return TogetherProvider(api_key=os.getenv("TOGETHERAI_API_KEY"))
    else:
        raise ValueError(f"Unsupported model: {model}")
