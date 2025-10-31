import streamlit as st
import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.llm_client import LLMClient, get_available_models
from utils.search_tools import WebSearchTool, format_search_results

st.title("Echo Bot")

st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(255,255,255,0.2), rgba(255,255,255,0.5)),url("https://png.pngtree.com/background/20230426/original/pngtree-mt-fuji-in-fall-image-of-autumn-forest-with-lake-picture-image_2481764.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        
    </style>
    """,
    unsafe_allow_html=True
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user", avatar="🧑")
    st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    
    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="📚"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
with st.sidebar:
        st.header("Settings")

        # Model selection
        available_models = get_available_models()
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            index=0,
            help="Choose the language model to use"
        )

        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.6,
            step=0.1,
            help="Controls randomness in responses"
        )

        # Max tokens
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=4000,
            value=1000,
            step=50,
            help="Maximum length of response"
        )

        with st.spinner("Initializing model..."):
            st.session_state.llm_client = LLMClient(
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens
        )
        st.markdown(selected_model)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()  # Refresh the app to show cleared chat

        message_count = len(st.session_state.messages)
        st.markdown(f"**Messages exchanged:** {message_count}")