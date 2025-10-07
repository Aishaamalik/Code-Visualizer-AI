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
	st.session_state.autoplay_interval = 3.0


# ----- Utility Functions -----
def build_highlighted_code_block(code_text: str, current_line: Optional[int], language: str) -> str:
	"""Return HTML that renders code lines with the current line highlighted.

	Uses <mark> for the active line index (1-based). Falls back gracefully if line
	number is out of range. Shows context around the current line.
	"""
	lines = code_text.splitlines()
	html_lines: List[str] = []
	
	# Show context around current line (3 lines before and after)
	start_idx = max(0, (current_line or 1) - 4)
	end_idx = min(len(lines), (current_line or 1) + 3)
	
	for idx in range(start_idx, end_idx):
		line_num = idx + 1
		line_content = lines[idx]
		
		# Create line number with padding
		line_num_str = f"{line_num:3d}"
		
		if current_line == line_num:
			html_lines.append(f"<div style='background:#fff3bf;border-left:4px solid #ff6b6b;padding-left:8px;'><span style='color:#666;margin-right:12px;'>{line_num_str}</span><code>{escape_html(line_content)}</code></div>")
		else:
			html_lines.append(f"<div><span style='color:#999;margin-right:12px;'>{line_num_str}</span><code>{escape_html(line_content)}</code></div>")
	
	# Wrap with a container mimicking code styling
	html = (
		"<div style=\"font-family:Menlo,Consolas,monospace;font-size:13px;"
		"line-height:1.4;border:1px solid #ddd;border-radius:8px;"
		"padding:16px;background:#f8f9fa;overflow-x:auto;box-shadow:0 2px 4px rgba(0,0,0,0.1);\">"
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
st.set_page_config(page_title="Code Visualizer AI", page_icon="🧠", layout="centered")

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

st.subheader("2) Visualization")
if st.session_state.analysis:
	st.write("**Summary:**")
	st.write(st.session_state.analysis.get("summary", ""))

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
		# Show different button text based on current state
		button_text = "⏸️ Pause" if st.session_state.playing else "▶️ Play"
		if st.button(button_text, use_container_width=True):
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

	# Code highlight with context
	st.markdown("**📍 Code Execution (Current line highlighted):**")
	st.markdown(
		build_highlighted_code_block(st.session_state.code, current_line, st.session_state.language),
		unsafe_allow_html=True,
	)

	# Enhanced Details Section
	with st.expander("📋 Step Details", expanded=True):
		# Operation and Context
		col1, col2 = st.columns([2, 1])
		with col1:
			st.markdown(f"**🔧 Operation:** {current.get('operation', 'Unknown')}")
			st.markdown(f"**📝 Explanation:** {current.get('explanation', 'No explanation available')}")
		with col2:
			st.markdown(f"**📍 Execution Context:** {current.get('execution_context', 'Main execution')}")
			if current.get('next_action'):
				st.markdown(f"**⏭️ Next:** {current.get('next_action', '')}")
		
		# Control Flow Information
		if current.get('control_flow'):
			st.markdown(f"**🔄 Control Flow:** {current.get('control_flow', '')}")
		
		# Main Information Grid
		cols = st.columns([1, 1, 1, 1])
		
		with cols[0]:
			st.markdown("**📊 Variables & Types**")
			variables = current.get("variables", {})
			if variables:
				for var_name, var_value in variables.items():
					if var_name.endswith('_type'):
						continue  # Skip type entries, we'll show them with main variables
					var_type = variables.get(f"{var_name}_type", "unknown")
					st.markdown(f"`{var_name}`: `{var_value}` ({var_type})")
			else:
				st.write("No variables")
		
		with cols[1]:
			st.markdown("**📞 Call Stack**")
			call_stack = current.get("call_stack", [])
			if call_stack:
				for i, func in enumerate(call_stack):
					indent = "  " * i
					st.write(f"{indent}→ {func}")
			else:
				st.write("Main execution")
		
		with cols[2]:
			st.markdown("**💾 Memory State**")
			memory_state = current.get("memory_state", {})
			if memory_state:
				heap = memory_state.get("heap", {})
				stack = memory_state.get("stack", [])
				if heap:
					st.markdown("**Heap:**")
					for obj_id, details in heap.items():
						st.write(f"  {obj_id}: {details}")
				if stack:
					st.markdown("**Stack:**")
					st.write(f"  {', '.join(stack)}")
			else:
				st.write("No memory details")
		
		with cols[3]:
			st.markdown("**📤 Outputs**")
			outputs = current.get("outputs", "")
			if outputs:
				st.code(outputs)
			else:
				st.write("No output")
	
	# Data Structures Visualization
	data_structures = current.get("data_structures", {})
	if data_structures:
		with st.expander("🏗️ Data Structures", expanded=False):
			for ds_name, ds_info in data_structures.items():
				st.markdown(f"**{ds_name}:**")
				if isinstance(ds_info, dict):
					if "elements" in ds_info:
						# List visualization
						elements = ds_info.get("elements", [])
						current_index = ds_info.get("index", 0)
						length = ds_info.get("length", len(elements))
						
						# Create a visual representation
						if elements:
							st.markdown(f"Length: {length}, Current Index: {current_index}")
							# Show elements with current index highlighted
							element_display = []
							for i, elem in enumerate(elements):
								if i == current_index:
									element_display.append(f"[{i}]: **{elem}** ←")
								else:
									element_display.append(f"[{i}]: {elem}")
							st.write(" | ".join(element_display))
						else:
							st.write("Empty list")
					
					elif "keys" in ds_info and "values" in ds_info:
						# Dictionary visualization
						keys = ds_info.get("keys", [])
						values = ds_info.get("values", [])
						current_key = ds_info.get("current_key", "")
						
						st.markdown(f"Keys: {keys}")
						st.markdown(f"Values: {values}")
						if current_key:
							st.markdown(f"Current Key: **{current_key}**")
					
					else:
						# Generic structure
						st.json(ds_info)
				else:
					st.write(ds_info)
	
	# Variable History (if available)
	if current_idx > 0 and current.get("variables"):
		with st.expander("📈 Variable Changes", expanded=False):
			st.markdown("**Current Variables:**")
			current_vars = current.get("variables", {})
			if len(steps) > current_idx:
				prev_vars = steps[current_idx - 1].get("variables", {})
				
				for var_name, var_value in current_vars.items():
					if var_name.endswith('_type'):
						continue
					prev_value = prev_vars.get(var_name, "undefined")
					if prev_value != var_value:
						st.markdown(f"`{var_name}`: `{prev_value}` → `{var_value}` 🔄")
					else:
						st.markdown(f"`{var_name}`: `{var_value}` (unchanged)")
			else:
				for var_name, var_value in current_vars.items():
					if not var_name.endswith('_type'):
						st.markdown(f"`{var_name}`: `{var_value}`")

	# Auto-advance when playing - use a placeholder for smoother updates
	auto_placeholder = st.empty()
	
	if st.session_state.playing and total_steps > 0:
		with auto_placeholder.container():
			st.info("🔄 Auto-playing... Click Play/Pause to stop")
		
		# Check if enough time has passed for the next step
		now = time.time()
		if now - st.session_state.last_tick >= st.session_state.autoplay_interval:
			# Move to next step
			new_step = (current_idx + 1) % total_steps
			st.session_state.current_step = new_step
			st.session_state.last_tick = now
			
			# Force a rerun to update the display
			st.rerun()
		else:
			# If still playing but not time yet, schedule a rerun in a short time
			time.sleep(0.1)
			st.rerun()
	else:
		# Clear the auto-play message when paused
		auto_placeholder.empty()
