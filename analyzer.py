import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from json_repair import repair_json

load_dotenv()


def get_groq_api_key() -> Optional[str]:
	key = os.getenv("GROQ_API_KEY")
	if key:
		return key
	try:
		from streamlit.runtime.secrets import AttrDict  # type: ignore
		import streamlit as st  # lazy import
		secrets_obj = getattr(st, "secrets", None)
		if secrets_obj is not None:
			return secrets_obj.get("GROQ_API_KEY", None)  # type: ignore[attr-defined]
	except Exception:
		return None
	return None


def build_prompts(language: str, code_text: str) -> Tuple[str, str]:
	system_prompt = (
		"You are Code Visualizer AI. Analyze the provided code and produce a strict JSON "
		"object describing execution step by step. The JSON MUST parse with Python json.loads. "
		"Do not include any comments, explanations, or markdown fences in the output."
	)
	user_prompt = f"""
Language: {language}
Code:
<CODE>
{code_text}
</CODE>

Return ONLY valid JSON with this exact structure (no prose, no code fences):
{{
  "language": "{language}",
  "summary": "One-paragraph summary of what the code does",
  "steps": [
    {{
      "step": 1,
      "line": 1,  
      "operation": "Short title of what happens",
      "explanation": "Plain-English explanation of this step",
      "variables": {{ "varName": "value or description" }},
      "call_stack": ["main"],
      "outputs": ""
    }}
  ]
}}

Rules:
- "line" must be an integer line number or null if unknown.
- Use only double quotes for all strings.
- Do not include trailing commas.
- Do not include additional keys beyond those listed.
"""
	return system_prompt, user_prompt


def _strip_code_fences(text: str) -> str:
	if text.strip().startswith("```"):
		# remove first and last triple backticks blocks
		return re.sub(r"^```[a-zA-Z0-9]*\n|\n```$", "", text.strip())
	return text


def _remove_trailing_commas(json_like: str) -> str:
	# Remove trailing commas before } or ]
	return re.sub(r",\s*(?=[}\]])", "", json_like)


def _extract_json_block(text: str) -> Optional[str]:
	start = text.find("{")
	end = text.rfind("}")
	if start != -1 and end != -1 and end > start:
		return text[start : end + 1]
	return None


def safe_json_loads(content: str) -> Dict[str, Any]:
	# First try direct parse
	try:
		return json.loads(content)
	except Exception:
		pass
	# Strip code fences
	clean = _strip_code_fences(content)
	candidate = _extract_json_block(clean) or clean
	# Use json_repair to fix broken JSON
	repaired = repair_json(candidate)
	return json.loads(repaired)


def analyze_code_with_llm(
	code_text: str,
	language: str,
	model_name: str = "llama-3.1-8b-instant",
	temperature: float = 0.4,
	max_tokens: int = 1000,
) -> Dict[str, Any]:
	api_key = get_groq_api_key()
	if not api_key:
		raise RuntimeError("Missing GROQ_API_KEY. Set it in .env or Streamlit secrets to analyze code.")

	llm = ChatGroq(
		api_key=api_key,
		model=model_name,
		temperature=temperature,
		max_tokens=max_tokens,
	)

	system_prompt, user_prompt = build_prompts(language, code_text)
	messages = [("system", system_prompt), ("user", user_prompt)]
	response = llm.invoke(messages)
	content = response.content if hasattr(response, "content") else str(response)

	result = safe_json_loads(content)

	result.setdefault("language", language)
	result.setdefault("summary", "")
	steps = result.get("steps", []) or []
	normalized_steps: List[Dict[str, Any]] = []
	for i, step in enumerate(steps, start=1):
		normalized_steps.append(
			{
				"step": step.get("step", i),
				"line": step.get("line", None),
				"operation": step.get("operation", ""),
				"explanation": step.get("explanation", ""),
				"variables": step.get("variables", {}),
				"call_stack": step.get("call_stack", []),
				"outputs": step.get("outputs", ""),
			}
		)
	result["steps"] = normalized_steps
	return result
