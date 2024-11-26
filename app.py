
import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
import tempfile
from dotenv import load_dotenv
load_dotenv()


# Load environment variables
load_dotenv()

# Initialize session states
if 'current_groq_key_index' not in st.session_state:
    st.session_state.current_groq_key_index = 0
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Llama-3.1-70b"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None


def create_llm(model_config):
    if model_config["class"] == ChatGroq:
        try:
            return model_config["class"](**model_config["params"])
        except Exception as e:
            if "quota exceeded" in str(e).lower() or "rate limit" in str(e).lower():
                next_key = get_next_groq_key()
                model_config["params"]["api_key"] = next_key
                return model_config["class"](**model_config["params"])
            raise e
    return model_config["class"](**model_config["params"])

# Define model configurations
MODEL_CONFIGS = {
    "Llama-3.1-70b": {
        "class": ChatGroq,
        "params": {
            "api_key": os.environ.get("GROQ_API_KEYS"),
            "model_name": "llama-3.1-70b-versatile",
            "temperature": 0.3
        }
    },
    "GPT-3.5-turbo": {
        "class": ChatOpenAI,
        "params": {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.3
        }
    },
    "GPT-4o-mini": {
        "class": ChatOpenAI,
        "params": {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "model_name": "gpt-4o-mini",
            "temperature": 0.3
        }
    }
}

# Function to scrape website content
def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        return ""

# Function to process documents
def process_documents(uploaded_files):
    documents = []
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name

        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == '.csv':
            loader = CSVLoader(temp_file_path)
        else:  # Assume text file for other extensions
            loader = TextLoader(temp_file_path)

        documents.extend(loader.load())
        os.unlink(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(documents)
    return split_documents

# Function to process websites and create vectorstore
def process_websites(urls):
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    for url in urls:
        content = scrape_website(url)
        if content:
            chunks = text_splitter.split_text(content)
            texts.extend(chunks)
    
    if texts:
        vectorstore = FAISS.from_texts(texts, embeddings)
        st.session_state.vectorstore = vectorstore
        st.success("Websites processed and vectorstore created successfully!")
    else:
        st.error("No content could be extracted from the provided URLs.")

# Streamlit UI
st.title("GreeneStep Helpbot Agent")

# Model selection
col1, col2 = st.columns([2, 4])
with col1:
    selected_model = st.selectbox(
        "Select Model:",
        options=list(MODEL_CONFIGS.keys()),
        index=list(MODEL_CONFIGS.keys()).index(st.session_state.selected_model)
    )

# Update session state when model changes
if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model

# File upload for training
st.subheader("Train RAG with Files")
uploaded_files = st.file_uploader("Upload your training data", accept_multiple_files=True, type=['pdf', 'csv', 'txt'])

# Website input for training
st.subheader("Train RAG with Websites")
websites = st.text_area("Enter website URLs (one per line) for training:", height=100)

# Train button
train_button = st.button("Train RAG")

if train_button:
    if uploaded_files:
        documents = process_documents(uploaded_files)
        embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_documents(documents, embeddings)
        st.success("Documents processed and vectorstore created successfully!")
    
    if websites:
        urls = [url.strip() for url in websites.split('\n') if url.strip()]
        process_websites(urls)

# Rest of the code (chat interface, question processing, etc.)
# Load embeddings and vector database
embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
if st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})
else:
    new_db = FAISS.load_local("Store", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_type="mmr", search_kwargs={"k": 10})

# System prompts
instruction_to_system = """
Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do not answer the question just formulate it if needed and otherwise return it as it is.
"""

qa_system_prompt = """
You are a GreeneStep AI Helpbot Agent for question-answering, helping prospects, partners, employees and customers of GreeneStep on GreeneStep Products, Solutions, Services, Pricing, Processes based on User Manual Guides, Knowledgebase Articles, How to do articles and videos, FAQs and Other GreeneStep Website and Product Landing Pages. Use the following pieces of retrieved context to answer the question: {context}
GIFT=GreeneStep Intelligent Forecasting Technology
GeS/GS = GreeneStep
BOF = Back Office
CFX = Cloud Front Web Access Portal
GBS = GreeneStep Bussiness Suite
If the question is out of context or your knowledge base then simply say I don't know the answer, ask me anything related to GeS Help. Do not generate your own answer.
"""

# Create prompts and chains
question_maker_prompt = ChatPromptTemplate.from_messages([
    ("system", instruction_to_system),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

model_config = MODEL_CONFIGS[st.session_state.selected_model]
llm = create_llm(model_config)
output_parser = StrOutputParser()
question_chain = question_maker_prompt | llm | output_parser

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return question_chain
    else:
        return input["question"]

retriever_chain = RunnablePassthrough.assign(
    context=contextualized_question | retriever
)

rag_chain = (
    retriever_chain | qa_prompt | llm
)

def process_rag_chain(question, chat_history):
    try:
        return rag_chain.invoke({"question": question, "chat_history": chat_history})
    except Exception as e:
        if "quota exceeded" in str(e).lower() or "rate limit" in str(e).lower():
            global llm
            model_config = MODEL_CONFIGS[st.session_state.selected_model]
            llm = create_llm(model_config)
            return rag_chain.invoke({"question": question, "chat_history": chat_history})
        raise e

# Input and submit
input_col, submit_col = st.columns([5, 2])
with input_col:
    question = st.text_input("Your Question:", placeholder="Please enter your question here......")
with submit_col:
    submit_button = st.button("Submit", use_container_width=True)

# Process question
if submit_button:
    if question:
        ai_response = process_rag_chain(question, st.session_state.chat_history)
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response.content
        })
        st.session_state.last_question = question
    else:
        st.warning("Please enter a question.")

# Display chat history
if st.session_state.chat_history:
    combined_text = ""
    for entry in st.session_state.chat_history:
        role = "Q" if entry["role"] == "user" else "A"
        combined_text += f"{role}: {entry['content']}\n\n"
    st.text_area("Conversation", value=combined_text, height=300)

# Control buttons
col3, col4, col5 = st.columns([2, 2, 2])
with col3:
    clear_button = st.button("Clear History", use_container_width=True)
with col4:
    delete_button = st.button("Delete Previous", use_container_width=True)
with col5:
    retry_button = st.button("Retry Last", use_container_width=True)

# Button actions
if clear_button:
    st.session_state.chat_history = []
    st.session_state.last_question = ""
if delete_button:
    if len(st.session_state.chat_history) >= 2:
        st.session_state.chat_history.pop()
        st.session_state.chat_history.pop()
    elif len(st.session_state.chat_history) == 1:
        st.session_state.chat_history.pop()
if retry_button:
    if st.session_state.last_question:
        question = st.session_state.last_question
        ai_response = process_rag_chain(question, st.session_state.chat_history)
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response.content
        })

