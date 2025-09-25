"""
Streamlit application for handling chat requests and managing conversation history.
This application uses the MCPAgent to interact with the Mistral AI model.
"""
import os
import getpass
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from mcp_use import MCPClient, MCPAgent

load_dotenv()


def get_agent_and_client(config_path: str):
    """
    Initialize and return the MCP client and agent using the given config path.
    """
    if "MISTRAL_API_KEY" not in os.environ:
        os.environ["MISTRAL_API_KEY"] = getpass.getpass(
            "Enter your Mistral API key: ")

    mcp_client = MCPClient.from_dict(config_path)
    llm = ChatMistralAI(
        model="mistral-medium-2505",
        callbacks=[StreamingStdOutCallbackHandler()],
        max_tokens=4096,
        temperature=0.7,
        streaming=True
    )
    agent = MCPAgent(
        llm=llm,
        client=mcp_client,
        max_steps=15,
        memory_enabled=True
    )
    return agent, mcp_client


# Initialize session state variables
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "client" not in st.session_state:
    st.session_state.client = None

# Config for multi-server setup
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

# Streamlit UI
st.title("Playwright and GitHub Assistant Tool")

# Tool selection
tool = st.sidebar.selectbox(
    "Select Tool",
    ["browser", "github", "atlassian"],
    key="tool"
)

# Map tool name to config file path
config_map = {
    "browser": "./browser_mcp.json",
    "github": "./github_mcp.json",
    "atlassian": "./atlassian_mcp.json",
}

# Initialize agent if not already done
if st.session_state.agent is None:
    st.session_state.agent, st.session_state.client = get_agent_and_client(
        multi_server)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("What's your message?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Clear conversation if requested
    if prompt.strip().lower() == "clear":
        st.session_state.agent.clear_conversation_history()
        st.session_state.messages = []
        with st.chat_message("assistant"):
            st.write("Conversation history cleared.")
    else:
        instructions = """
Use the Playwright MCP server tool **only** if the user's prompt contains any Playwright-specific context or mentions of Playwright (e.g., browser automation, testing with Playwright, playwright scripts, etc.).

If no Playwright-related context is found, then default to using the GitHub MCP server tool to access repositories for user 'navazkhanai'.

If no direct list command is available for GitHub, use the GitHub API through the available MCP server tools.
"""
        # Prepend instructions to the user message
        prompt_with_instructions = f"{instructions}\n\n{prompt}"

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Progressing..."):
                try:
                    async def get_response():
                        return await st.session_state.agent.run(prompt_with_instructions)

                    response = asyncio.run(get_response())
                    response_placeholder.write(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    # Reset agent on error
                    st.session_state.agent, st.session_state.client = get_agent_and_client(
                        multi_server)

# Add a button to close all sessions
if st.sidebar.button("Close All Sessions"):
    try:
        if st.session_state.client:
            asyncio.run(st.session_state.client.close_all_sessions())
            st.sidebar.success("All sessions closed successfully.")
            # Reset agent and client
            st.session_state.agent = None
            st.session_state.client = None
    except Exception as e:
        st.sidebar.error(f"Error closing sessions: {str(e)}")
