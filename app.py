# from langchain_community.vectorstores import FAISS
# from langchain.prompts.prompt import PromptTemplate
# # from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA
# from langchain.chains.llm import LLMChain
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# import gradio as gr
# # from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv
# load_dotenv()



# # ...............Saving and Loading................................
# embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # save_directory = "Storage amul"
# # index.save_local(save_directory)
# new_db = FAISS.load_local("embeddings_output_portfolio",embeddings,allow_dangerous_deserialization=True)


# # ...............Defining Retriever................................
# retriever = new_db.as_retriever(search_type="similarity",search_kwargs={"k":3})



# # ...............Prompt............................................
# template = """ Start answering the question like a Pro chatbot and the question based on the provided context. if you do not found relavent result just say "This information is not available with me". 

# Context: {context}
# Question : {question}

# Helpful Answer :
# """
# prompt = PromptTemplate(template=template,input_variables=["context","question"])



# # ..............Define LLM Model....................................
# # llm = Ollama(model="llama3.1:8b",temperature=0.6).
# # Define LLM
# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model_name="llama-3.1-70b-versatile"
# )

# # ..............Define qa_Chain........................................
# llm_chain = LLMChain(
#                   llm=llm, 
#                   prompt=prompt, 
#                   callbacks=None, 
#                   verbose=True)

# document_prompt = PromptTemplate(
#     input_variables=["page_content", "source"],
#     template="Context:\ncontent:{page_content}\nsource:{source}",
# )

# combine_documents_chain = StuffDocumentsChain(
#                   llm_chain=llm_chain,
#                   document_variable_name="context",
#                   document_prompt=document_prompt,
#                   callbacks=None,
#               )


# qa = RetrievalQA(
#                   combine_documents_chain=combine_documents_chain,
#                   verbose=True,
#                   retriever=retriever,
#                   return_source_documents=True,
#               )


# # print(qa("What is described inside?")["result"])

# def respond(question,history):
#     return qa(question)["result"]


# gr.ChatInterface(
#     respond,
#     chatbot=gr.Chatbot(height=200),
#     textbox=gr.Textbox(placeholder="Ask me question related to Amul", container=False, scale=7),
#     title="Amul Portfolio"

# ).launch(share = True)



############### streamlit



import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

# ...............Load and Configure FAISS Database................
embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
new_db = FAISS.load_local("embeddings_output_portfolio", embeddings, allow_dangerous_deserialization=True)

# ...............Define Retriever................................
retriever = new_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ...............Prompt Template................................
template = """ Start answering the question like a Pro chatbot and the question based on the provided context. If you do not find relevant results, just say "This information is not available with me. Also give precise answer not too much and it should be accurate" 

Context: {context}
Question : {question}

Helpful Answer :
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# ..............Define LLM Model................................
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-70b-versatile"
)

# ..............Define qa_Chain..................................
llm_chain = LLMChain(
    llm=llm, 
    prompt=prompt, 
    callbacks=None, 
    verbose=True
)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
    callbacks=None,
)

qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    verbose=True,
    retriever=retriever,
    return_source_documents=True,
)

# Streamlit Chat Interface
st.title("Amul's Portfolio Agent")

if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# User input
question = st.chat_input("Ask me a question about Amul")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    # Call the QA system
    response = qa(question)["result"]
    # Append response to chat history
    st.session_state.history.append({"role": "assistant", "content": response})
    # Display the response
    with st.chat_message("assistant"):
        st.markdown(response)

