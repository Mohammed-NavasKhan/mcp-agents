"""
Flask application for handling chat requests and managing conversation history.
This application uses the MCPAgent to interact with the Mistral AI model.
"""
import os
import getpass
import asyncio
from threading import Lock
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from mcp_use import MCPClient, MCPAgent


session_agents = {}
session_lock = Lock()

load_dotenv()

app = Flask(__name__)
CORS(app)


def get_agent_and_client(config_path: str):
    """
    Initialize and return the MCP client and agent using the given config path.
    """
    if "MISTRAL_API_KEY" not in os.environ:
        os.environ["MISTRAL_API_KEY"] = getpass.getpass(
            "Enter your Mistral API key: ")

    # client = MCPClient.from_config_file(config_path)
    client = MCPClient.from_dict(config_path)
    llm = ChatMistralAI(model="mistral-medium-2505")
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=True,
    )
    return agent, client


@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint to handle chat messages and manage conversation history."""
    data = request.get_json()
    messages = data.get("messages")
    tool = data.get("tool", None)
    print(f"Received messages: {messages}, tool: {tool}")
    session_id = data.get("session_id")
    # Map tool name to config file path
    config_map = {
        "browser": "./browser_mcp.json",
        "github": "./github_mcp.json",
        "atlassian": "./atlassian_mcp.json",
    }
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "'messages' must be a non-empty list."}), 400

    last_msg = messages[-1] if messages else None
    if not last_msg or last_msg.get("role") != "user" or not last_msg.get("content") or not last_msg["content"].strip():
        return jsonify({"error": "Last message must be from user and contain non-empty content."}), 400

    tool_key = tool.lower() if isinstance(tool, str) else None
    mcp_config_path = config_map.get(tool_key)
    multi_server = {
        "mcpServers": {
            "playwright": {
                "command": "npx",
                "args": ["@playwright/mcp@latest"],
            },
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": "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"
                }
            }
        }
    }
    print(f"Using MCP config path: {mcp_config_path}")
    if not mcp_config_path:
        return jsonify({"error": f"Unsupported tool: {tool}"}), 400

    # Use session_id to persist agent per session
    if not session_id:
        return jsonify({"error": "session_id is required for conversation continuity."}), 400
    with session_lock:
        if session_id not in session_agents:
            # agent, _ = get_agent_and_client(mcp_config_path)
            agent, _ = get_agent_and_client(multi_server)
            session_agents[session_id] = agent
        else:
            agent = session_agents[session_id]

    if last_msg["content"].strip().lower() == "clear":
        agent.clear_conversation_history()
        return jsonify({"response": "Conversation history cleared."})

    instructions = """
Use the Playwright MCP server tool **only** if the user's prompt contains any Playwright-specific context or mentions of Playwright (e.g., browser automation, testing with Playwright, playwright scripts, etc.).

If no Playwright-related context is found, then default to using the GitHub MCP server tool to access repositories for user 'navazkhanai'.

If no direct list command is available for GitHub, use the GitHub API through the available MCP server tools.
"""
    # Prepend instructions to the user message
    prompt_with_instructions = f"{instructions}\n\n{last_msg['content']}"
    try:
        response = asyncio.run(agent.run(prompt_with_instructions))
        return jsonify({"response": response})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except AttributeError as e:
        return jsonify({"error": str(e)}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Endpoint to close all active sessions."""
    try:
        _, client = get_agent_and_client("./browser_mcp.json")
        asyncio.run(client.close_all_sessions())
        return jsonify({"response": "All sessions closed."})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except AttributeError as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Use PORT env var if set, else default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
