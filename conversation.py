import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import time

load_dotenv()


os.environ['HUGGING_FACE'] = os.getenv('HUGGING_FACE')

embeddings = HuggingFaceEmbeddings(model='all-MiniLM-L6-v2')

## streamlit
st.title('Conversational RAG with PDF uplaods and chat History')
st.write("Uplaod Pdf's and chat with their content")

with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=groq_api_key)
uploaded_file = None

if 'store' not in st.session_state:
    st.session_state.store = {}

if groq_api_key:
    session_id = st.text_input('Session ID', value='Default_session')
    uploaded_file = st.file_uploader('Chookse A PDF file', type='pdf', accept_multiple_files=False)
else:
    st.error('Please enter the GROQ API KEY and press Enter to proceed')
    

if uploaded_file:
    doc = []
    for uploaded_flies in uploaded_file:
        temppdf = f"./temp.pdf"
        with open(temppdf, 'wb') as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

    doc.extend(PyPDFLoader(temppdf).load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(doc)
    retriever = Chroma.from_documents(docs, embeddings).as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', contextualize_q_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)

    ## Answer Question Prompt

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)


    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )

    user_input = st.text_input('Your question:')
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {
                'input': user_input,
            },
            config={'configurable': {'session_id': session_id}}
        )

        #st.write(st.session_state.store)
        st.write("Assistant:", response['answer'])
        #st.write("Chat History:", session_history.messages)