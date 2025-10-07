import os
import time
import json
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from analyzer import analyze_code_with_llm, get_groq_api_key


# ----- Environment & Config -----
load_dotenv()


# ----- Session State Initialization -----
if "code" not in st.session_state:
	st.session_state.code = ""
if "language" not in st.session_state:
	st.session_state.language = "python"
if "analysis" not in st.session_state:
	st.session_state.analysis = None  # type: Optional[Dict[str, Any]]
if "steps" not in st.session_state:
	st.session_state.steps = []  # type: List[Dict[str, Any]]
if "current_step" not in st.session_state:
	st.session_state.current_step = 0
if "playing" not in st.session_state:
	st.session_state.playing = False
if "last_tick" not in st.session_state:
	st.session_state.last_tick = time.time()
if "autoplay_interval" not in st.session_state:
	st.session_state.autoplay_interval = 0.8


# ----- Utility Functions -----
def build_highlighted_code_block(code_text: str, current_line: Optional[int], language: str) -> str:
	"""Return HTML that renders code lines with the current line highlighted.

	Uses <mark> for the active line index (1-based). Falls back gracefully if line
	number is out of range.
	"""
	lines = code_text.splitlines()
	html_lines: List[str] = []
	for idx, line in enumerate(lines, start=1):
		if current_line == idx:
			html_lines.append(f"<div style='background:#fff3bf'><code>{escape_html(line)}</code></div>")
		else:
			html_lines.append(f"<div><code>{escape_html(line)}</code></div>")
	# Wrap with a container mimicking code styling
	html = (
		"<div style=\"font-family:Menlo,Consolas,monospace;font-size:13px;"
		"line-height:1.4;border:1px solid #eee;border-radius:6px;"
		"padding:12px;background:#fafafa;overflow-x:auto;\">"
		+ "\n".join(html_lines)
		+ "</div>"
	)
	return html


def escape_html(text: str) -> str:
	return (
		text.replace("&", "&amp;")
		.replace("<", "&lt;")
		.replace(">", "&gt;")
		.replace('"', "&quot;")
		.replace("'", "&#39;")
	)


# ----- UI -----
st.set_page_config(page_title="Code Visualizer AI", page_icon="🧠", layout="wide")

st.title("Code Visualizer AI 🧠➡️🖼️")
st.caption(
	"Interactive visualization of code execution using Groq Llama-3.1-8B-Instant via LangChain."
)

# Simple status about API key, without sidebar
api_key = get_groq_api_key()
if api_key:
	st.info("API ready.")
else:
	st.warning("Set GROQ_API_KEY in .env or Streamlit secrets to enable analysis.")

col_left, col_right = st.columns([1, 1])

with col_left:
	st.subheader("1) Paste Your Code")
	st.session_state.language = st.selectbox(
		"Language",
		options=["python", "javascript", "cpp", "java", "csharp", "go", "rust", "ruby"],
		index=["python", "javascript", "cpp", "java", "csharp", "go", "rust", "ruby"].index(
			st.session_state.language
		)
		if st.session_state.language in [
			"python",
			"javascript",
			"cpp",
			"java",
			"csharp",
			"go",
			"rust",
			"ruby",
		]
		else 0,
	)
	st.session_state.code = st.text_area(
		"Code",
		value=st.session_state.code,
		height=300,
		placeholder="Paste any code here to visualize its execution...",
	)

	analyze_clicked = st.button("Analyze Code", type="primary", disabled=not bool(st.session_state.code))
	if analyze_clicked:
		try:
			analysis = analyze_code_with_llm(
				st.session_state.code,
				st.session_state.language,
			)
			st.session_state.analysis = analysis
			st.session_state.steps = analysis.get("steps", [])
			st.session_state.current_step = 0
			st.session_state.playing = False
			st.success("Analysis complete.")
		except Exception as e:
			st.session_state.analysis = None
			st.session_state.steps = []
			st.session_state.current_step = 0
			st.session_state.playing = False
			st.error(f"Analysis failed: {e}")

with col_right:
	st.subheader("2) Visualization")
	steps: List[Dict[str, Any]] = st.session_state.steps
	if not steps:
		st.info("Paste code and click Analyze to see step-by-step visualization.")
	else:
		total_steps = len(steps)
		current_idx = max(0, min(st.session_state.current_step, total_steps - 1))
		current = steps[current_idx]
		current_line = current.get("line", None)

		# Controls
		ctrl_prev, ctrl_play, ctrl_next, ctrl_reset = st.columns([1, 1, 1, 1])
		with ctrl_prev:
			if st.button("◀️ Prev", use_container_width=True):
				st.session_state.current_step = (current_idx - 1) % total_steps
				st.session_state.playing = False
				st.rerun()
		with ctrl_play:
			if st.button("⏯ Play/Pause", use_container_width=True):
				st.session_state.playing = not st.session_state.playing
				st.session_state.last_tick = time.time()
				st.rerun()
		with ctrl_next:
			if st.button("Next ▶️", use_container_width=True):
				st.session_state.current_step = (current_idx + 1) % total_steps
				st.session_state.playing = False
				st.rerun()
		with ctrl_reset:
			if st.button("🔁 Reset", use_container_width=True):
				st.session_state.current_step = 0
				st.session_state.playing = False
				st.rerun()

		st.progress((current_idx + 1) / total_steps, text=f"Step {current_idx + 1} / {total_steps}")

		# Code highlight
		st.markdown("Current line highlighted in yellow:")
		st.markdown(
			build_highlighted_code_block(st.session_state.code, current_line, st.session_state.language),
			unsafe_allow_html=True,
		)

		# Details
		with st.expander("Step details", expanded=True):
			st.write(f"Operation: {current.get('operation', '')}")
			st.write(current.get("explanation", ""))
			cols = st.columns([1, 1, 1])
			with cols[0]:
				st.markdown("**Variables**")
				st.json(current.get("variables", {}))
			with cols[1]:
				st.markdown("**Call stack**")
				st.write(", ".join(current.get("call_stack", [])))
			with cols[2]:
				st.markdown("**Output**")
				st.code(current.get("outputs", ""))

		# Auto-advance when playing
		if st.session_state.playing and total_steps > 0:
			now = time.time()
			if now - st.session_state.last_tick >= st.session_state.autoplay_interval:
				st.session_state.current_step = (current_idx + 1) % total_steps
				st.session_state.last_tick = now
				st.rerun()
