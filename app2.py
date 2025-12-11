# Streamlit for UI dev
import streamlit as st

# watsonx interface
from langchain_ibm import WatsonxLLM

# --- 1. SETUP UI FIRST ---
# Move this to the top so the user sees the app loading immediately
st.title('Ask watsonx')

# --- 2. DEFINE CREDENTIALS & PARAMETERS ---
# (Replace with your actual keys, but keep them secret!)
credentials = {
    "url": "https://eu-gb.ml.cloud.ibm.com",
    "apikey": "2r5A0gcx3CzU13CpbSnDkymwuG3L7nvOU_fGqTkZUkX6"
}

parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 500,
    "min_new_tokens": 1,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 1,
}

# --- 3. INITIALIZE LLM (UNCOMMENTED) ---
try:
    llm = WatsonxLLM(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url=credentials["url"],
        apikey=credentials["apikey"],
        project_id="4807c8e5-4890-4a76-8bf5-a3a7ac61e9ef",
        params=parameters
    )
except Exception as e:
    st.error(f"Failed to load Watsonx model. Check your API Key and Project ID.\nError: {e}")
    st.stop() # Stop execution safely if model fails

# --- 4. SESSION STATE ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- 5. DISPLAY HISTORY ---
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# --- 6. CHAT INPUT LOOP ---
prompt = st.chat_input('Pass Your Prompt here')

if prompt:
    # Display prompt
    st.chat_message('user').markdown(prompt)
    # Store user prompt state
    st.session_state.messages.append({'role':'user', 'content':prompt})

    # Send prompt to LLM (Using invoke instead of direct call is safer in new versions)
    # Note: 'llm(prompt)' is deprecated in LangChain v0.1+. Use 'llm.invoke(prompt)'
    response = llm.invoke(prompt)
    
    # Show LLM response
    st.chat_message('assistant').markdown(response)
    # Store LLM response in state
    st.session_state.messages.append(
        {'role':'assistant', 'content':response}
    )

# --- 7. OTHER IMPORTS ---
# Keep these at the bottom or top as preferred, but ensure they don't block the UI
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings