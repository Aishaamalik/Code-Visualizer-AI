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
		"For the summary, provide a comprehensive analysis that includes: "
		"1) Clear explanation of the code's purpose and main functionality, "
		"2) Detailed breakdown of each function and its role, "
		"3) Algorithm logic and approach used, "
		"4) Data flow and how variables are used throughout, "
		"5) Expected outputs and behavior, "
		"6) Key programming concepts demonstrated, "
		"7) Time/space complexity analysis if applicable, "
		"8) Educational insights about the code structure. "
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
  "summary": "Comprehensive analysis including: 1) Code purpose and main functionality, 2) Function definitions and their roles, 3) Algorithm logic and approach, 4) Data flow and variable usage, 5) Expected outputs and behavior, 6) Key concepts demonstrated, 7) Complexity analysis if applicable",
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

SUMMARY REQUIREMENTS:
- Write a comprehensive, educational summary (3-5 paragraphs)
- Explain the code's purpose and main functionality clearly
- Break down each function and explain its specific role
- Describe the algorithm logic and approach used
- Explain data flow and variable usage patterns
- Detail expected outputs and behavior
- Highlight key programming concepts demonstrated
- Include complexity analysis (time/space) where applicable
- Provide educational insights about code structure and design patterns
- Make it suitable for learning and understanding the code thoroughly
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
	max_tokens: int = 6000,
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


def _build_debugger_prompts(language: str, code_text: str) -> Tuple[str, str]:
	system_prompt = (
		"You are an expert debugging assistant. Identify syntax, runtime, and logic errors in code. "
		"Explain the root cause in plain language and propose safe, actionable fixes. "
		"Provide a fully corrected version of the code that applies those fixes. "
		"Return ONLY strict JSON matching the required schema."
	)
	user_prompt = f"""
Language: {language}
Code:
<CODE>
{code_text}
</CODE>

Analyze potential issues WITHOUT executing the code. Provide a JSON object with this exact structure:
{{
  "issues": [
    {{
      "type": "syntax|runtime|logic|style|performance",
      "line": 0,
      "title": "Short human-readable title",
      "explanation": "Clear explanation of the problem and its impact",
      "suggestion": "Concrete fix with example code or steps"
    }}
  ],
  "corrected_code": "FULL corrected code with all safe fixes applied"
}}

Rules:
- If line cannot be determined, set line to 0.
- Prefer minimal, safe fixes. Do NOT invent APIs. Provide targeted suggestions.
- Use only double quotes in JSON.
"""
	return system_prompt, user_prompt


def analyze_errors_with_llm(
	code_text: str,
	language: str,
	model_name: str = "llama-3.1-8b-instant",
	temperature: float = 0.1,
	max_tokens: int = 3000,
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

	system_prompt, user_prompt = _build_debugger_prompts(language, code_text)
	messages = [("system", system_prompt), ("user", user_prompt)]
	response = llm.invoke(messages)
	content = response.content if hasattr(response, "content") else str(response)

	parsed = safe_json_loads(content)
	issues = parsed.get("issues", []) or []
	# Normalize shape
	normalized: List[Dict[str, Any]] = []
	for issue in issues:
		normalized.append(
			{
				"type": str(issue.get("type", "logic")),
				"line": int(issue.get("line", 0) or 0),
				"title": issue.get("title", "Potential issue"),
				"explanation": issue.get("explanation", ""),
				"suggestion": issue.get("suggestion", ""),
			}
		)
	corrected_code = parsed.get("corrected_code", "") or ""
	return {"issues": normalized, "corrected_code": corrected_code}


def _build_complexity_prompts(language: str, code_text: str) -> Tuple[str, str]:
	system_prompt = (
		"You are an algorithms and complexity expert. Estimate precise time and space complexity for functions. "
		"Identify dominant terms and provide Big-O, with notes on best/average/worst if relevant. "
		"Detect loops and recursion, and model their growth. Return ONLY strict JSON."
	)
	user_prompt = f"""
Language: {language}
Code:
<CODE>
{code_text}
</CODE>

Provide complexity analysis as JSON with this exact structure:
{{
  "functions": [
    {{
      "name": "function_name",
      "time_complexity": "O(n log n)",
      "space_complexity": "O(n)",
      "notes": "Short justification and dominant factors",
      "loops": [{{"location": "line 23", "complexity": "O(n)", "explanation": "for loop over n items"}}],
      "recursions": [{{"location": "line 45", "recurrence": "T(n)=2T(n/2)+n", "solution": "O(n log n)"}}]
    }}
  ]
}}

Rules:
- If names are unknown, infer reasonable names like "main".
- Use only double quotes in JSON.
"""
	return system_prompt, user_prompt


def analyze_complexity_with_llm(
	code_text: str,
	language: str,
	model_name: str = "llama-3.1-8b-instant",
	temperature: float = 0.1,
	max_tokens: int = 3500,
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

	system_prompt, user_prompt = _build_complexity_prompts(language, code_text)
	messages = [("system", system_prompt), ("user", user_prompt)]
	response = llm.invoke(messages)
	content = response.content if hasattr(response, "content") else str(response)

	parsed = safe_json_loads(content)
	functions = parsed.get("functions", []) or []
	normalized: List[Dict[str, Any]] = []
	for fn in functions:
		normalized.append(
			{
				"name": fn.get("name", "main"),
				"time_complexity": fn.get("time_complexity", "O(1)"),
				"space_complexity": fn.get("space_complexity", "O(1)"),
				"notes": fn.get("notes", ""),
				"loops": fn.get("loops", []),
				"recursions": fn.get("recursions", []),
			}
		)
	return {"functions": normalized}
