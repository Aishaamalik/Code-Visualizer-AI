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
		"You are Code Visualizer AI, an expert at analyzing code execution step by step. "
		"Your task is to create a detailed, accurate visualization of how code executes. "
		"Analyze every line, every operation, every variable change, and every control flow decision. "
		"Be extremely precise about variable values, memory states, and execution flow. "
		"Return ONLY valid JSON that can be parsed with Python json.loads(). "
		"No markdown, no explanations outside the JSON structure."
	)
	user_prompt = f"""
Language: {language}
Code:
<CODE>
{code_text}
</CODE>

Create a detailed step-by-step execution analysis. Return ONLY valid JSON with this exact structure:
{{
  "language": "{language}",
  "summary": "Detailed summary explaining what the code does, its purpose, and main functionality",
  "steps": [
    {{
      "step": 1,
      "line": 1,
      "operation": "Detailed operation name (e.g., 'Variable Declaration', 'Function Call', 'Conditional Check', 'Loop Iteration')",
      "explanation": "Detailed explanation of what happens at this step, including the reasoning and context",
      "variables": {{ 
        "varName": "current_value", 
        "varName_type": "data_type",
        "varName_address": "memory_location_if_relevant"
      }},
      "call_stack": ["function_name", "nested_function"],
      "outputs": "Any console output, return values, or side effects",
      "memory_state": {{ 
        "heap": {{ "object_id": "object_details" }},
        "stack": ["local_variables"]
      }},
      "control_flow": "branch_taken or loop_condition or exception_handling",
      "data_structures": {{
        "list_name": {{ "elements": [1, 2, 3], "index": 0, "length": 3 }},
        "dict_name": {{ "keys": ["key1"], "values": ["value1"], "current_key": "key1" }}
      }},
      "execution_context": "What part of the program is currently executing",
      "next_action": "What will happen in the next step"
    }}
  ]
}}

CRITICAL RULES:
1. Track EVERY variable assignment, function call, conditional check, and loop iteration
2. Show the EXACT values of variables at each step
3. Include data type information for variables
4. Track function calls and return values accurately
5. Show control flow decisions (if/else branches, loop conditions)
6. For data structures (lists, dicts, objects), show their current state
7. Include memory management details when relevant
8. Be precise about line numbers - each executable line should have a step
9. Show the execution context (which function, loop iteration, etc.)
10. Use only double quotes, no trailing commas, valid JSON syntax
11. Make explanations educational and detailed for learning purposes
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
	temperature: float = 0.2,
	max_tokens: int = 4000,
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
				"memory_state": step.get("memory_state", {}),
				"control_flow": step.get("control_flow", ""),
				"data_structures": step.get("data_structures", {}),
				"execution_context": step.get("execution_context", ""),
				"next_action": step.get("next_action", ""),
			}
		)
	result["steps"] = normalized_steps
	return result
