import os
import subprocess
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain.schema import SystemMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]

# Safe commands only
SAFE_COMMANDS = ['echo', 'type', 'mkdir', 'cd', 'dir', 'pip install', 'python']

@tool
def run_command(command):
    """Executes safe system commands."""
    if not any(command.strip().lower().startswith(safe) for safe in SAFE_COMMANDS):
        return "⛔ Unsafe command detected. Execution blocked."
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        result = e.output
    return result.strip()

@tool
def write_file(file_path, content):
    """Writes content to a file."""
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        return f"✅ Content written to {file_path}"
    except Exception as e:
        return f"❌ Error writing to file: {e}"

@tool
def read_file(file_path):
    """Reads content from a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"❌ Error reading file: {e}"

def run_command_int_helper(command):
    if not any(command.strip().lower().startswith(safe) for safe in SAFE_COMMANDS):
        return "⛔ Unsafe command detected. Execution blocked."
    try:
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        result = e.output
    return result.strip()

@tool
def install_package(package_name):
    """Installs a Python package."""
    return run_command_int_helper(f"pip install {package_name}")


initial_prompt = f"""
You are a helpful AI Assistant specialized in resolving user queries using available tools.
Work on start, plan, action, observe, and output modes.

Rules:
- Wite all your code in AI_code folder if not present then create one.
- Always output in JSON format.
- Perform one action at a time and wait for the observation before proceeding.
- Carefully analyze the user query and break it into small logical steps if needed.
- Only use the available tools.
- Never said user to copy paste the code or to create any folder o file.

Output JSON Format:
{{
    "step": "string",         # plan, action, observe, or output
    "content": "string",      # Thought process if step is plan or output
    "function": "string",     # Function name if step is action
    "input": "any"            # Input parameters for the function if step is action
}}

Available Tools:
- run_command: Execute safe system commands.
- write_file: Write content to a file.
- read_file: Read content from a file.
- install_package: Install a Python package.

---

Example 1:
User Query: Create a Python file named app.py and write a program that prints "Hello, World!".

Output:
{{ "step": "plan", "content": "The user wants to create a Python file with specific content." }}
{{ "step": "action", "function": "write_file", "input": {{"file_path": "app.py", "content": "print('Hello, World!')"}} }}
{{ "step": "observe", "output": "✅ Content written to app.py" }}
{{ "step": "output", "content": "The file app.py has been created and populated successfully." }}

---
 
Example 2:
User Query: Install the numpy library.

Output:
{{ "step": "plan", "content": "The user wants to install a Python package called numpy." }}
{{ "step": "action", "function": "install_package", "input": "numpy" }}
{{ "step": "observe", "output": "Successfully installed numpy" }}
{{ "step": "output", "content": "The numpy package has been installed successfully." }}

---

Example 3:
User Query: Read the contents of the file app.py.

Output:
{{ "step": "plan", "content": "The user wants to read the contents of an existing file." }}
{{ "step": "action", "function": "read_file", "input": "app.py" }}
{{ "step": "observe", "output": "print('Hello, World!')" }}
{{ "step": "output", "content": "The contents of app.py were read successfully." }}

---

Example 4:
User Query: Make a folder named 'projects'.

Output:
{{ "step": "plan", "content": "The user wants to create a new directory." }}
{{ "step": "action", "function": "run_command", "input": "mkdir projects" }}
{{ "step": "observe", "output": "Directory 'projects' created successfully." }}
{{ "step": "output", "content": "Folder 'projects' has been created." }}
"""

# Bind model and tools
llm = init_chat_model(model_provider="openai", model="gpt-4o")
llm_with_tool = llm.bind_tools(tools=[run_command, write_file, read_file, install_package])

# Node for language model reasoning
def chatbot(state: State):
    system_prompt = SystemMessage(content=initial_prompt)
    message = llm_with_tool.invoke([system_prompt] + state["messages"])
    return {"messages": [message]}

# Node for executing tools
tool_node = ToolNode(tools=[run_command, write_file, read_file, install_package])

# Build the graph
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# Exportable function to create the chat graph
def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
