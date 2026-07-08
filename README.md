# 🤖 Advanced Conversational RAG Engine with History-Aware Retrieval

A production-grade, stateful **Retrieval-Augmented Generation (RAG)** digital assistant built using **LangChain**, **Streamlit**, and **Groq Cloud API**. This application handles long-context text ingestion from live web structures, embeds data using local transformer models, persists semantic data points into a local vector index, and applies sequential prompt-rewriting algorithms to guarantee continuous conversational memory without LLM context degradation.

---

## 📐 Enterprise Architecture Flow

```mermaid
graph TD
    %% Ingestion Pipeline
    subgraph Document Ingestion Pipeline
        A[WebBaseLoader / BeautifulSoup] -->|Extract Raw HTML/Text| B[RecursiveCharacterTextSplitter]
        B -->|1000-char semantic chunks| C[HuggingFace Inference Engine]
        C -->|all-MiniLM-L6-v2 Embeddings| D[(Chroma Vector DB)]
    end

    %% Inference Pipeline
    subgraph Execution & Contextualization Layer
        E[User Dynamic Chat Input] -->|Raw Query| F{Contextualization Decision}
        G[(st.session_state.store)] -->|Historical Message Buffer| F
        F -->|Prompt Interception| H[Llama-3.3-70b-Versatile]
        H -->|De-contextualized Standalone Question| I[Vector Similarity Search]
        D -->|Top-K Document Retrieval| I
    end

    %% Synthesis Pipeline
    subgraph Response Generation Layer
        I -->|Retrieved Context + Standalone Q| J[Document Stuffing Chain]
        J -->|Refined Execution Prompt| K[Llama-3.3-70b Synthesis Engine]
        K -->|Deterministic, Grounded Output| L[Streamlit Chat UI Frontend]
    end

    style D fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px





## 📐 How It Works (Architecture)

```mermaid
graph TD
    A[Web Loader] -->|Scrapes Web Content| B[Text Splitter]
    B -->|Breaks Text into Chunks| C[HuggingFace Embeddings]
    C -->|Converts Text to Numbers| D[(Chroma Vector DB)]
    E[User Query + Chat History] -->|Rewrites Query| F[Groq Llama 3.3]
    D -->|Finds Match| F
    F -->|Generates Answer| G[Streamlit Chat UI]
