# watsonx interface
from langchain_ibm import WatsonxLLM

# Streamlit for UI dev
import streamlit as st



# Credentials dictionary
creds = {
    #'apikey':'7P_8YeFi4DJTKGvSsUMMZD7gX2pdWYGTM4LB1cL8n2Mw',
    #'url': 'https://eu-gb.ml.cloud.ibm.com'
}

# Create LLM through Langchain
parameters = {
    #"decoding_method": "sample",
    #"max_new_tokens": 500,
    #"min_new_tokens": 1,
    #"temperature": 0.7,
    #"top_k": 50,
    #"top_p": 1,
}



# Initialise the new Llama 3.3 model

llm = WatsonxLLM(
    #model_id="meta-llama/llama-3-3-70b-instruct",
    #url="https://us-south.ml.cloud.ibm.com",
    #project_id="4807c8e5-4890-4a76-8bf5-a3a7ac61e9ef",
    #params=parameters
)


# App title
st.title('Ask watsonx')

# Session state message variable (holds old messages)
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Prompt input template displaying prompts
prompt = st.chat_input('Pass Your Prompt here')

# When user hits enter
if prompt:
    # Display prompt
    st.chat_message('user').markdown(prompt)
    # Store user prompt state
    st.session_state.messages.append({'role':'user', 'content':prompt})

    # Send prompt to LLM
    response = llm(prompt)
    # Show LLM response
    st.chat_message('assistant').markdown(response)
    # Store LLM response in state
    st.session_state.messages.append(
        {'role':'assistant', 'content':response}
    )



# Langchain dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.indexes import VectorstoreIndexCreator
from langchain_classic.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

