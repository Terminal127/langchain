#!/usr/bin/env python3

import os
import sys
import uuid
import json
import requests
import subprocess
import readline
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
import warnings

warnings.filterwarnings("ignore", message="Convert_system_message_to_human will be deprecated!")

# Global variables
chatmap = {}
session_id = None
HISTORY_FILE = "chat_history.json"
COMMAND_HISTORY_FILE = os.path.expanduser("~/.langchain_chat_history")

def setup_readline():
    """Setup readline for command history and arrow key support"""
    try:
        # Enable tab completion
        readline.parse_and_bind("tab: complete")
        
        # Set history file
        if os.path.exists(COMMAND_HISTORY_FILE):
            readline.read_history_file(COMMAND_HISTORY_FILE)
        
        # Set maximum history length
        readline.set_history_length(1000)
        
        # Enable arrow keys and command editing
        readline.parse_and_bind("set editing-mode emacs")
        readline.parse_and_bind("set show-all-if-ambiguous on")
        readline.parse_and_bind("set completion-ignore-case on")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not setup readline: {e}")

def save_readline_history():
    """Save command history to file"""
    try:
        readline.write_history_file(COMMAND_HISTORY_FILE)
    except Exception as e:
        print(f"⚠️  Warning: Could not save command history: {e}")

# Define tools for the agent
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely. Use this for math operations."""
    try:
        # Basic safety check - only allow numbers, operators, and parentheses
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"

@tool
def get_weather(city: str) -> str:
    """Get current weather information for a city."""
    try:
        # Using a free weather API (replace with your preferred service)
        # This is a mock implementation - you'd need to sign up for a real API
        return f"Mock weather data for {city}: Sunny, 22°C, Light breeze"
    except Exception as e:
        return f"Error getting weather: {str(e)}"

@tool
def file_operations(operation: str, filename: str, content: str = "") -> str:
    """Perform file operations: read, write, or list files.
    Operations: 'read', 'write', 'list'
    """
    try:
        if operation == "read":
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    return f.read()
            return f"File {filename} not found"
        
        elif operation == "write":
            with open(filename, 'w') as f:
                f.write(content)
            return f"Content written to {filename}"
        
        elif operation == "list":
            files = os.listdir(filename if filename else ".")
            return "\n".join(files)
        
        else:
            return "Invalid operation. Use 'read', 'write', or 'list'"
    
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def run_command(command: str) -> str:
    """Execute a safe shell command and return its output."""
    try:
        # Whitelist of safe commands
        safe_commands = ['ls', 'pwd', 'date', 'whoami', 'echo', 'cat', 'head', 'tail']
        cmd_parts = command.split()
        
        if not cmd_parts or cmd_parts[0] not in safe_commands:
            return "Error: Command not allowed for security reasons"
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    
    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_chat_history_summary() -> str:
    """Get a summary of the current chat session history including recent questions and responses."""
    global chatmap, session_id
    
    if session_id not in chatmap or not chatmap[session_id].messages:
        return "No chat history found in current session."
    
    history = chatmap[session_id].messages
    
    # Get recent messages (last 10 interactions)
    recent_messages = history[-20:] if len(history) > 20 else history
    
    summary = "Recent chat history:\n"
    for i, msg in enumerate(recent_messages):
        role = "User" if msg.type == "human" else "Assistant"
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        summary += f"{i+1}. {role}: {content}\n"
    
    return summary

@tool
def task_planner(user_request: str) -> str:
    """Plan out the steps needed to complete a complex user request. Use this when the user asks for multiple things to be done."""
    # This tool helps the AI think through complex requests step by step
    return f"Planning steps for: {user_request}\n" \
           f"1. Identify all requested actions\n" \
           f"2. Determine which tools are needed\n" \
           f"3. Execute tools in proper sequence\n" \
           f"4. Verify all tasks are completed\n" \
           f"Remember: Actually use the tools, don't just describe what you would do!"

# List of available tools
tools = [
    get_current_time,
    calculate,
    get_weather,
    file_operations,
    run_command,
    get_chat_history_summary,
    task_planner
]

def load_chat_history():
    """Load chat history from file"""
    global chatmap, session_id
    
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
                session_id = data.get('session_id', str(uuid.uuid4()))
                
                # Reconstruct chat history
                if 'messages' in data:
                    history = InMemoryChatMessageHistory()
                    for msg in data['messages']:
                        if msg['type'] == 'human':
                            history.add_message(HumanMessage(content=msg['content']))
                        else:
                            history.add_message(AIMessage(content=msg['content']))
                    chatmap[session_id] = history
        except Exception as e:
            print(f"⚠️  Error loading history: {e}")
            session_id = str(uuid.uuid4())
    else:
        session_id = str(uuid.uuid4())

def save_chat_history():
    """Save chat history to file"""
    global chatmap, session_id
    
    try:
        data = {'session_id': session_id, 'messages': []}
        
        if session_id in chatmap:
            for msg in chatmap[session_id].messages:
                data['messages'].append({
                    'type': msg.type,
                    'content': msg.content,
                    'timestamp': datetime.now().isoformat()
                })
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"⚠️  Error saving history: {e}")

def setup_model():
    """Initialize the Gemini model"""
    api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyDsi82MHuNMwZyUoJ5q6xN8yd9Q4yBw5gM")
    
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=api_key,
        convert_system_message_to_human=True
    )
    return model

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a session"""
    global chatmap
    if session_id not in chatmap:
        chatmap[session_id] = InMemoryChatMessageHistory()
    return chatmap[session_id]

def setup_agent_executor(model):
    """Setup the LangChain agent executor with tools and chat history"""
    # Create agent prompt with scratchpad
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an intelligent assistant with access to powerful tools. You MUST use tools for ANY action you mention.\n\n"
         
         "🎯 CORE RULE: If you say you will do something, YOU MUST DO IT using the appropriate tool.\n\n"
         
         "📋 AVAILABLE TOOLS & WHEN TO USE:\n"
         "• get_current_time → When time/date is needed or mentioned\n"
         "• calculate → For ANY mathematical expression, even simple ones\n" 
         "• get_weather → When weather information is requested\n"
         "• file_operations → For reading files, writing/saving content, or listing directories\n"
         "• run_command → For system info (whoami for username, ls for files, pwd for location, etc.)\n"
         "• get_chat_history_summary → When asked about previous conversation\n"
         "• task_planner → For complex multi-step requests\n\n"
         
         "🔥 MANDATORY ACTIONS:\n"
         "1. Use multiple tools in sequence for complex tasks\n\n"
         
         "💡 SMART BEHAVIOR:\n"
         "• Never say 'I will do X' without actually doing X\n\n"
         
         "📝 RESPONSE FORMAT:\n"
         "• Start with '#'\n"
         "• Be concise (2-3 sentences)\n"
         "• Execute tools immediately when needed\n"
         "• Confirm completion of actions\n\n"
         
         "⚠️ NEVER say you 'cannot determine' something if a tool can help - USE THE TOOL!"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create the agent
    agent = create_tool_calling_agent(model, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5  # Allow more iterations for complex tasks
    )
    
    # Wrap with message history
    agent_with_history = RunnableWithMessageHistory(
        runnable=agent_executor,
        get_session_history=get_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    
    return agent_with_history

def get_response(agent_executor, question: str) -> str:
    """Get response from the AI agent"""
    global session_id
    try:
        response = agent_executor.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )
        return response.get("output", "No response generated")
    except Exception as e:
        return f"Error: {str(e)}"

def print_welcome():
    """Print welcome message"""
    global session_id
    print("\n" + "="*60)
    print("🤖 LANGCHAIN AGENT CHAT APPLICATION")
    print("="*60)
    print("✨ Powered by Google Gemini 2.0 Flash with Tools")
    print(f"📱 Session ID: {session_id[:8]}...")
    print("📝 Type your message and press Enter")
    print("⌨️  Use ↑/↓ arrow keys to navigate command history")
    print("🛠️  Available Tools: time, calculator, weather, files, commands, search")
    print("💡 Commands: /help, /history, /clear, /quit, /tools")
    print("="*60 + "\n")

def print_help():
    """Print help message"""
    print("\n📚 AVAILABLE COMMANDS:")
    print("  /help     - Show this help message")
    print("  /history  - Show chat history")
    print("  /clear    - Clear chat history")
    print("  /quit     - Exit the application")
    print("  /session  - Show current session info")
    print("  /new      - Start new chat session")
    print("  /tools    - Show available tools")
    print()

def print_tools():
    """Print available tools"""
    print("\n🛠️  AVAILABLE TOOLS:")
    print("  get_current_time      - Get current date and time")
    print("  calculate             - Perform mathematical calculations")
    print("  get_weather           - Get weather information for a city")
    print("  file_operations       - Read, write, or list files")
    print("  run_command           - Execute safe shell commands")
    print("  get_chat_history_summary - Access recent chat history")
    print("  task_planner          - Plan out steps for complex requests")
    print("\n💡 Just ask naturally, like:")
    print("  - 'What time is it?'")
    print("  - 'Calculate 15 * 23 + 7'")
    print("  - 'What's the weather in London?'")
    print("  - 'List files in current directory'")
    print("  - 'What questions have I asked?'")
    print("  - 'Create a story and save it to a file'")
    print()

def print_history():
    """Print chat history"""
    global chatmap, session_id
    history = chatmap.get(session_id)
    if not history or not history.messages:
        print("📝 No chat history found.")
        return
    
    print("\n📜 CHAT HISTORY:")
    print("-" * 50)
    for i, msg in enumerate(history.messages, 1):
        timestamp = datetime.now().strftime("%H:%M:%S")
        role = "🧑 You" if msg.type == "human" else "🤖 AI"
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"{i}. [{timestamp}] {role}: {content}")
    print("-" * 50 + "\n")

def clear_history():
    """Clear chat history"""
    global chatmap, session_id
    if session_id in chatmap:
        chatmap[session_id].clear()
    # Also clear the file
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    print("🗑️ Chat history cleared!\n")

def new_session():
    """Start a new chat session"""
    global session_id
    # Save current session before creating new one
    save_chat_history()
    session_id = str(uuid.uuid4())
    print(f"🆕 New session started: {session_id[:8]}...\n")

def print_session_info():
    """Print current session information"""
    global chatmap, session_id
    history = chatmap.get(session_id)
    msg_count = len(history.messages) if history else 0
    print(f"\n📊 SESSION INFO:")
    print(f"   Session ID: {session_id}")
    print(f"   Messages: {msg_count}")
    print(f"   Model: Gemini 2.0 Flash (Agent)")
    print(f"   Tools: {len(tools)} available")
    print()

def main():
    """Main function to run the chat application"""
    global session_id
    
    # Check if Google API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY environment variable not set.")
        print("   Using hardcoded key (not recommended for production)")
        print("   Set it with: export GOOGLE_API_KEY='your-api-key-here'")
        print()
    
    # Setup readline for command history and arrow keys
    setup_readline()
    
    # Load existing chat history
    load_chat_history()
    
    # Setup model and agent executor
    model = setup_model()
    agent_executor = setup_agent_executor(model)
    
    print_welcome()
    
    try:
        while True:
            # Get user input with readline support
            try:
                user_input = input("🧑 You: ").strip()
            except EOFError:
                # Handle Ctrl+D
                print("\n👋 Goodbye! Thanks for chatting!")
                break
            
            # Handle empty input
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command == '/quit' or command == '/exit':
                    save_chat_history()  # Save before exiting
                    save_readline_history()  # Save command history
                    print("👋 Goodbye! Thanks for chatting!")
                    break
                elif command == '/help':
                    print_help()
                elif command == '/history':
                    print_history()
                elif command == '/clear':
                    clear_history()
                elif command == '/session':
                    print_session_info()
                elif command == '/new':
                    new_session()
                elif command == '/tools':
                    print_tools()
                else:
                    print("❌ Unknown command. Type /help for available commands.\n")
                continue
            
            # Get AI response
            print("🤖 AI: ", end="", flush=True)
            response = get_response(agent_executor, user_input)
            print(response)
            print()
            
            # Save history after each interaction
            save_chat_history()
            
    except KeyboardInterrupt:
        print("\n\n👋 Chat interrupted. Goodbye!")
        save_chat_history()
        save_readline_history()
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        save_chat_history()
        save_readline_history()
    finally:
        sys.exit(0)

if __name__ == "__main__":
    main()