import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


NO_THINKING_SYSTEM_PROMPT = (
    "You are a concise assistant. Do not output reasoning, analysis, or thinking process. "
    "Return only the final answer."
)


class LocalQwenGenerator:
    def __init__(
        self,
        model_id="Qwen/Qwen3.5-27B-FP8",
        device="cuda",
        trust_remote_code=True,
    ):
        self.model_id = model_id
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.tokenizer = None
        self.model = None

    def _load(self):
        if self.tokenizer is not None and self.model is not None:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda" and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=self.trust_remote_code,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
                trust_remote_code=self.trust_remote_code,
            ).to("cpu")

        self.model.eval()

    def chat(
        self,
        user_prompt,
        system_prompt=None,
        max_new_tokens=196,
        temperature=0.7,
        top_p=0.9,
        enable_thinking=False,
    ):
        self._load()

        messages = []
        if not enable_thinking:
            if system_prompt is None:
                system_prompt = NO_THINKING_SYSTEM_PROMPT
            else:
                system_prompt = NO_THINKING_SYSTEM_PROMPT + " " + system_prompt
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        prompt_text = user_prompt
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except ValueError:
                prompt_text = user_prompt

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            except RuntimeError as error:
                message = str(error)
                if "device-side assert" in message or "CUDA error" in message:
                    raise RuntimeError(
                        "CUDA generation failed. Restart the notebook kernel and retry generation."
                    ) from error
                raise

        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate(self, prompt, max_new_tokens=196, temperature=0.7, top_p=0.9, enable_thinking=False):
        return self.chat(
            user_prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
