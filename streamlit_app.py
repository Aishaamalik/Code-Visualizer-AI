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
st.set_page_config(
	page_title="Code Visualizer AI", 
	page_icon="🧠", 
	layout="wide",
	initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling
st.markdown("""
<style>
	/* Main container styling */
	.main .block-container {
		padding-top: 2rem;
		padding-bottom: 2rem;
		max-width: 1200px;
	}
	
	/* Header styling */
	.header-container {
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		padding: 2rem;
		border-radius: 15px;
		margin-bottom: 2rem;
		color: white;
		text-align: center;
		box-shadow: 0 8px 32px rgba(0,0,0,0.1);
	}
	
	.header-title {
		font-size: 2.5rem;
		font-weight: 700;
		margin-bottom: 0.5rem;
		background: linear-gradient(45deg, #fff, #f0f0f0);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
	}
	
	.header-subtitle {
		font-size: 1.1rem;
		opacity: 0.9;
		margin-bottom: 0;
	}
	
	/* Card styling */
	.card {
		background: white;
		padding: 1.5rem;
		border-radius: 12px;
		box-shadow: 0 4px 20px rgba(0,0,0,0.08);
		border: 1px solid #e1e5e9;
		margin-bottom: 1rem;
	}
	
	/* Control buttons styling */
	.control-button {
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		color: white;
		border: none;
		border-radius: 8px;
		padding: 0.5rem 1rem;
		font-weight: 600;
		transition: all 0.3s ease;
		box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
	}
	
	.control-button:hover {
		transform: translateY(-2px);
		box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
	}
	
	/* Progress bar styling */
	.progress-container {
		background: #f8f9fa;
		border-radius: 10px;
		padding: 1rem;
		margin: 1rem 0;
		border: 1px solid #e9ecef;
	}
	
	/* Info panels styling */
	.info-panel {
		background: #f8f9fa;
		border-radius: 10px;
		padding: 1.5rem;
		border-left: 4px solid #667eea;
		margin: 1rem 0;
	}
	
	.info-panel h3 {
		color: #495057;
		margin-bottom: 1rem;
		font-size: 1.2rem;
	}
	
	/* Variable display styling */
	.variable-item {
		background: white;
		padding: 0.75rem;
		border-radius: 8px;
		border: 1px solid #e9ecef;
		margin: 0.5rem 0;
		box-shadow: 0 2px 4px rgba(0,0,0,0.05);
	}
	
	/* Status indicators */
	.status-playing {
		background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
		color: white;
		padding: 0.5rem 1rem;
		border-radius: 20px;
		font-weight: 600;
		animation: pulse 2s infinite;
	}
	
	@keyframes pulse {
		0% { opacity: 1; }
		50% { opacity: 0.7; }
		100% { opacity: 1; }
	}
	
	/* Code block styling */
	.code-container {
		background: #1e1e1e;
		border-radius: 12px;
		padding: 1.5rem;
		box-shadow: 0 8px 32px rgba(0,0,0,0.2);
		border: 1px solid #333;
	}
	
	/* Section headers */
	.section-header {
		font-size: 1.5rem;
		font-weight: 600;
		color: #495057;
		margin: 2rem 0 1rem 0;
		border-bottom: 3px solid #667eea;
		padding-bottom: 0.5rem;
	}
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="header-container">
	<h1 class="header-title">🧠 Code Visualizer AI</h1>
	<p class="header-subtitle">Interactive step-by-step code execution visualization powered by AI</p>
</div>
""", unsafe_allow_html=True)

# API Status
api_key = get_groq_api_key()
if api_key:
	st.markdown("""
	<div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); 
	            color: white; padding: 0.75rem 1.5rem; border-radius: 10px; 
	            margin-bottom: 1rem; text-align: center; font-weight: 600;">
		✅ API Ready - Ready to analyze your code!
	</div>
	""", unsafe_allow_html=True)
else:
	st.markdown("""
	<div style="background: linear-gradient(135deg, #ff6b6b 0%, #ffa8a8 100%); 
	            color: white; padding: 0.75rem 1.5rem; border-radius: 10px; 
	            margin-bottom: 1rem; text-align: center; font-weight: 600;">
		⚠️ Set GROQ_API_KEY in .env or Streamlit secrets to enable analysis
	</div>
	""", unsafe_allow_html=True)

# Code Input Section
st.markdown('<h2 class="section-header">📝 Code Input</h2>', unsafe_allow_html=True)

# Create two columns for language selection and analyze button
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
	st.session_state.language = st.selectbox(
		"🔤 Programming Language",
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
		help="Select the programming language of your code"
	)

with col2:
	st.markdown("") # Spacer
	st.markdown("") # Spacer

with col3:
	analyze_clicked = st.button(
		"🚀 Analyze Code", 
		type="primary", 
		disabled=not bool(st.session_state.code),
		use_container_width=True,
		help="Start the AI-powered code analysis"
	)

# Code input area with better styling
st.session_state.code = st.text_area(
	"💻 Your Code",
	value=st.session_state.code,
	height=300,
	placeholder="""# Paste your code here to visualize its execution step by step

# Example:
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(5)
print(f"Fibonacci(5) = {result}")""",
	help="Enter your code to see how it executes step by step"
)

if analyze_clicked:
	with st.spinner("🧠 AI is analyzing your code..."):
		try:
			analysis = analyze_code_with_llm(
				st.session_state.code,
				st.session_state.language,
			)
			st.session_state.analysis = analysis
			st.session_state.steps = analysis.get("steps", [])
			st.session_state.current_step = 0
			st.session_state.playing = False
			
			# Success message with better styling
			st.markdown("""
			<div style="background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); 
			            color: white; padding: 1rem 1.5rem; border-radius: 10px; 
			            margin: 1rem 0; text-align: center; font-weight: 600;">
				🎉 Analysis Complete! Scroll down to see the visualization.
			</div>
			""", unsafe_allow_html=True)
		except Exception as e:
			st.session_state.analysis = None
			st.session_state.steps = []
			st.session_state.current_step = 0
			st.session_state.playing = False
			
			# Error message with better styling
			st.markdown(f"""
			<div style="background: linear-gradient(135deg, #ff6b6b 0%, #ffa8a8 100%); 
			            color: white; padding: 1rem 1.5rem; border-radius: 10px; 
			            margin: 1rem 0; text-align: center; font-weight: 600;">
				❌ Analysis failed: {e}
			</div>
			""", unsafe_allow_html=True)

# Visualization Section
st.markdown('<h2 class="section-header">🎬 Code Execution Visualization</h2>', unsafe_allow_html=True)

if st.session_state.analysis:
	# Summary Card
	summary = st.session_state.analysis.get("summary", "")
	if summary:
		st.markdown(f"""
		<div class="card">
			<h3 style="color: #495057; margin-bottom: 1rem;">📋 Code Summary</h3>
			<p style="font-size: 1.1rem; line-height: 1.6; color: #6c757d;">{summary}</p>
		</div>
		""", unsafe_allow_html=True)

steps: List[Dict[str, Any]] = st.session_state.steps
if not steps:
	st.markdown("""
	<div class="card" style="text-align: center; padding: 3rem;">
		<h3 style="color: #6c757d; margin-bottom: 1rem;">🚀 Ready to Visualize!</h3>
		<p style="font-size: 1.1rem; color: #6c757d;">Paste your code above and click "Analyze Code" to see step-by-step execution visualization.</p>
	</div>
	""", unsafe_allow_html=True)
else:
	total_steps = len(steps)
	current_idx = max(0, min(st.session_state.current_step, total_steps - 1))
	current = steps[current_idx]
	current_line = current.get("line", None)

	# Enhanced Controls Section
	st.markdown("""
	<div class="progress-container">
	""", unsafe_allow_html=True)
	
	# Progress bar with better styling
	progress_value = (current_idx + 1) / total_steps
	st.progress(progress_value, text=f"Step {current_idx + 1} of {total_steps}")
	
	# Control buttons with modern styling
	col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
	
	with col1:
		if st.button("⏮️ First", use_container_width=True, help="Go to first step"):
			st.session_state.current_step = 0
			st.session_state.playing = False
			st.rerun()
	
	with col2:
		if st.button("◀️ Prev", use_container_width=True, help="Previous step"):
			st.session_state.current_step = (current_idx - 1) % total_steps
			st.session_state.playing = False
			st.rerun()
	
	with col3:
		# Show different button text based on current state
		button_text = "⏸️ Pause" if st.session_state.playing else "▶️ Play"
		button_type = "secondary" if st.session_state.playing else "primary"
		if st.button(button_text, type=button_type, use_container_width=True, help="Play/Pause auto-advance"):
			st.session_state.playing = not st.session_state.playing
			st.session_state.last_tick = time.time()
			st.rerun()
	
	with col4:
		if st.button("Next ▶️", use_container_width=True, help="Next step"):
			st.session_state.current_step = (current_idx + 1) % total_steps
			st.session_state.playing = False
			st.rerun()
	
	with col5:
		if st.button("⏭️ Last", use_container_width=True, help="Go to last step"):
			st.session_state.current_step = total_steps - 1
			st.session_state.playing = False
			st.rerun()
	
	# Reset button
	if st.button("🔄 Reset to Beginning", use_container_width=True, help="Reset to first step"):
		st.session_state.current_step = 0
		st.session_state.playing = False
		st.rerun()
	
	st.markdown("</div>", unsafe_allow_html=True)
	
	# Status indicator
	if st.session_state.playing:
		st.markdown("""
		<div class="status-playing" style="text-align: center; margin: 1rem 0;">
			🎬 Auto-playing - Steps advance every 3 seconds
		</div>
		""", unsafe_allow_html=True)

	# Code Execution Display
	st.markdown("""
	<div class="card">
		<h3 style="color: #495057; margin-bottom: 1rem; display: flex; align-items: center;">
			📍 Code Execution
			<span style="margin-left: auto; font-size: 0.9rem; color: #6c757d;">
				Line {current_line if current_line else 'N/A'}
			</span>
		</h3>
	</div>
	""", unsafe_allow_html=True)
	
	st.markdown(
		build_highlighted_code_block(st.session_state.code, current_line, st.session_state.language),
		unsafe_allow_html=True,
	)

	# Main Information Display in Cards
	col1, col2 = st.columns([2, 1])
	
	with col1:
		# Step Details Card
		st.markdown(f"""
		<div class="card">
			<h3 style="color: #495057; margin-bottom: 1rem;">🔧 Step {current_idx + 1} Details</h3>
			<div style="margin-bottom: 1rem;">
				<strong style="color: #667eea;">Operation:</strong> {current.get('operation', 'Unknown')}
			</div>
			<div style="margin-bottom: 1rem;">
				<strong style="color: #667eea;">Explanation:</strong><br>
				<span style="color: #6c757d;">{current.get('explanation', 'No explanation available')}</span>
			</div>
		</div>
		""", unsafe_allow_html=True)
		
		# Variables Card
		variables = current.get("variables", {})
		if variables:
			st.markdown("""
			<div class="card">
				<h3 style="color: #495057; margin-bottom: 1rem;">📊 Variables & Types</h3>
			""", unsafe_allow_html=True)
			
			for var_name, var_value in variables.items():
				if var_name.endswith('_type'):
					continue
				var_type = variables.get(f"{var_name}_type", "unknown")
				st.markdown(f"""
				<div class="variable-item">
					<strong style="color: #667eea;">{var_name}</strong>: 
					<code style="background: #f8f9fa; padding: 0.2rem 0.4rem; border-radius: 4px;">{var_value}</code>
					<span style="color: #6c757d; font-size: 0.9rem;">({var_type})</span>
				</div>
				""", unsafe_allow_html=True)
			
			st.markdown("</div>", unsafe_allow_html=True)
	
	with col2:
		# Execution Context Card
		st.markdown(f"""
		<div class="card">
			<h3 style="color: #495057; margin-bottom: 1rem;">📍 Execution Context</h3>
			<div style="margin-bottom: 1rem;">
				<strong style="color: #667eea;">Context:</strong><br>
				<span style="color: #6c757d;">{current.get('execution_context', 'Main execution')}</span>
			</div>
		""", unsafe_allow_html=True)
		
		if current.get('control_flow'):
			st.markdown(f"""
			<div style="margin-bottom: 1rem;">
				<strong style="color: #667eea;">Control Flow:</strong><br>
				<span style="color: #6c757d;">{current.get('control_flow', '')}</span>
			</div>
			""", unsafe_allow_html=True)
		
		if current.get('next_action'):
			st.markdown(f"""
			<div style="margin-bottom: 1rem;">
				<strong style="color: #667eea;">Next Action:</strong><br>
				<span style="color: #6c757d;">{current.get('next_action', '')}</span>
			</div>
			""", unsafe_allow_html=True)
		
		st.markdown("</div>", unsafe_allow_html=True)
		
		# Call Stack Card
		call_stack = current.get("call_stack", [])
		st.markdown("""
		<div class="card">
			<h3 style="color: #495057; margin-bottom: 1rem;">📞 Call Stack</h3>
		""", unsafe_allow_html=True)
		
		if call_stack:
			for i, func in enumerate(call_stack):
				indent = "&nbsp;" * (i * 2)
				st.markdown(f"""
				<div style="margin: 0.5rem 0; color: #6c757d;">
					{indent}→ <strong style="color: #667eea;">{func}</strong>
				</div>
				""", unsafe_allow_html=True)
		else:
			st.markdown("""
			<div style="color: #6c757d;">Main execution</div>
			""", unsafe_allow_html=True)
		
		st.markdown("</div>", unsafe_allow_html=True)
	
	# Additional Information in Expandable Sections
	col3, col4 = st.columns([1, 1])
	
	with col3:
		# Memory State
		memory_state = current.get("memory_state", {})
		if memory_state:
			with st.expander("💾 Memory State", expanded=False):
				heap = memory_state.get("heap", {})
				stack = memory_state.get("stack", [])
				if heap:
					st.markdown("**Heap:**")
					for obj_id, details in heap.items():
						st.write(f"  {obj_id}: {details}")
				if stack:
					st.markdown("**Stack:**")
					st.write(f"  {', '.join(stack)}")
	
	with col4:
		# Outputs
		outputs = current.get("outputs", "")
		if outputs:
			with st.expander("📤 Outputs", expanded=False):
				st.code(outputs)
	
	# Data Structures Visualization
	data_structures = current.get("data_structures", {})
	if data_structures:
		with st.expander("🏗️ Data Structures", expanded=False):
			for ds_name, ds_info in data_structures.items():
				st.markdown(f"""
				<div class="card" style="margin-bottom: 1rem;">
					<h4 style="color: #495057; margin-bottom: 1rem;">{ds_name}</h4>
				""", unsafe_allow_html=True)
				
				if isinstance(ds_info, dict):
					if "elements" in ds_info:
						# List visualization
						elements = ds_info.get("elements", [])
						current_index = ds_info.get("index", 0)
						length = ds_info.get("length", len(elements))
						
						if elements:
							st.markdown(f"""
							<div style="margin-bottom: 1rem;">
								<strong style="color: #667eea;">Length:</strong> {length} | 
								<strong style="color: #667eea;">Current Index:</strong> {current_index}
							</div>
							""", unsafe_allow_html=True)
							
							# Show elements with current index highlighted
							st.markdown("**Elements:**")
							for i, elem in enumerate(elements):
								if i == current_index:
									st.markdown(f"""
									<div style="background: #fff3bf; padding: 0.5rem; border-radius: 4px; 
									            border-left: 4px solid #ff6b6b; margin: 0.25rem 0;">
										<strong>[{i}]: {elem}</strong> ← Current
									</div>
									""", unsafe_allow_html=True)
								else:
									st.markdown(f"""
									<div style="background: #f8f9fa; padding: 0.5rem; border-radius: 4px; 
									            margin: 0.25rem 0;">
										[{i}]: {elem}
									</div>
									""", unsafe_allow_html=True)
						else:
							st.write("Empty list")
					
					elif "keys" in ds_info and "values" in ds_info:
						# Dictionary visualization
						keys = ds_info.get("keys", [])
						values = ds_info.get("values", [])
						current_key = ds_info.get("current_key", "")
						
						st.markdown(f"""
						<div style="margin-bottom: 1rem;">
							<strong style="color: #667eea;">Keys:</strong> {keys}<br>
							<strong style="color: #667eea;">Values:</strong> {values}
						</div>
						""", unsafe_allow_html=True)
						
						if current_key:
							st.markdown(f"""
							<div style="background: #fff3bf; padding: 0.5rem; border-radius: 4px; 
							            border-left: 4px solid #ff6b6b;">
								<strong style="color: #667eea;">Current Key:</strong> {current_key}
							</div>
							""", unsafe_allow_html=True)
					
					else:
						# Generic structure
						st.json(ds_info)
				else:
					st.write(ds_info)
				
				st.markdown("</div>", unsafe_allow_html=True)
	
	# Variable History (if available)
	if current_idx > 0 and current.get("variables"):
		with st.expander("📈 Variable Changes", expanded=False):
			st.markdown("""
			<div class="card">
				<h4 style="color: #495057; margin-bottom: 1rem;">Variable Changes from Previous Step</h4>
			""", unsafe_allow_html=True)
			
			current_vars = current.get("variables", {})
			if len(steps) > current_idx:
				prev_vars = steps[current_idx - 1].get("variables", {})
				
				changes_found = False
				for var_name, var_value in current_vars.items():
					if var_name.endswith('_type'):
						continue
					prev_value = prev_vars.get(var_name, "undefined")
					if prev_value != var_value:
						changes_found = True
						st.markdown(f"""
						<div style="background: #e8f5e8; padding: 0.75rem; border-radius: 6px; 
						            border-left: 4px solid #28a745; margin: 0.5rem 0;">
							<strong style="color: #495057;">{var_name}:</strong> 
							<code style="background: #fff; padding: 0.2rem 0.4rem; border-radius: 3px;">{prev_value}</code> 
							→ 
							<code style="background: #fff; padding: 0.2rem 0.4rem; border-radius: 3px;">{var_value}</code>
							<span style="color: #28a745;">🔄 Changed</span>
						</div>
						""", unsafe_allow_html=True)
					else:
						st.markdown(f"""
						<div style="background: #f8f9fa; padding: 0.5rem; border-radius: 6px; 
						            margin: 0.25rem 0; color: #6c757d;">
							<strong>{var_name}:</strong> {var_value} (unchanged)
						</div>
						""", unsafe_allow_html=True)
				
				if not changes_found:
					st.markdown("""
					<div style="text-align: center; color: #6c757d; padding: 1rem;">
						No variable changes in this step
					</div>
					""", unsafe_allow_html=True)
			else:
				for var_name, var_value in current_vars.items():
					if not var_name.endswith('_type'):
						st.markdown(f"""
						<div style="background: #f8f9fa; padding: 0.5rem; border-radius: 6px; 
						            margin: 0.25rem 0;">
							<strong style="color: #667eea;">{var_name}:</strong> {var_value}
						</div>
						""", unsafe_allow_html=True)
			
			st.markdown("</div>", unsafe_allow_html=True)

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
