import os
import bs4
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Page Configuration (MUST be at the absolute top)
st.set_page_config(
    page_title="Conversational RAG Chatbot", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


groq_api_key = st.secrets["GROQ_API_KEY"]
# 2. Setup System Caching for Production Efficiency
@st.cache_resource
def initialize_rag_system():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    vectors = Chroma.from_documents(split_docs, embedding=embeddings)
    retreiver = vectors.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retreiver, contextualize_q_prompt)

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# ==========================
# Session State
# ==========================
if "store" not in st.session_state:
    st.session_state.store = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = "user_session"

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Initialize RAG Engine

rag_chain = initialize_rag_system()

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# ==========================
# Sidebar
# ==========================
with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;">
            <h1>🧠</h1>
            <h2>Knowledge Engine</h2>
            <p style="color:gray;">
            Conversational Retrieval-Augmented Generation
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("📄 Source Document")

    st.text_input(
        "Indexed URL",
        value="https://en.wikipedia.org/wiki/Artificial_intelligence",
        disabled=True,
    )

    st.divider()

    st.subheader("🚀 Tech Stack")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("LLM", "Llama 3.3")
        st.metric("Embeddings", "MiniLM")

    with col2:
        st.metric("Vector DB", "Chroma")
        st.metric("Framework", "LangChain")

    st.divider()

    st.success("🟢 System Ready")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.store = {}
        st.rerun()


# ==========================
# Hero Section
# ==========================
st.markdown(
"""
# 🧠 Conversational Knowledge Bot

Ask intelligent questions over grounded knowledge using a
**history-aware Retrieval-Augmented Generation (RAG)** pipeline powered by
**Groq + LangChain + ChromaDB**.
"""
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("LLM", "Llama 3.3")
c2.metric("Retriever", "History Aware")
c3.metric("Embeddings", "MiniLM")
c4.metric("Vector Store", "ChromaDB")

st.divider()



# Chat History

for message in st.session_state.messages:

    avatar = "🧑🏻" if message["role"] == "user" else "🤖"

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])



# Chat Input
user_query = st.chat_input(
    "Ask anything about the indexed document..."
)

if user_query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message("user", avatar="🧑🏻"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🤖"):

        with st.spinner("Searching knowledge base..."):

            try:

                response = conversational_rag_chain.invoke(
                    {"input": user_query},
                    config={
                        "configurable": {
                            "session_id": st.session_state.session_id
                        }
                    },
                )

                answer = response["answer"]

                st.markdown(answer)

                if "context" in response:

                    with st.expander("📚 Retrieved Context"):

                        for i, doc in enumerate(response["context"], start=1):

                            st.markdown(f"**Chunk {i}**")

                            st.info(doc.page_content[:500] + "...")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as e:

                st.error(f"❌ {e}")
