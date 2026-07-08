---

## 📐 How It Works (Architecture)

```mermaid
graph TD
    A[Web Loader] -->|Scrapes Web Content| B[Text Splitter]
    B -->|Breaks Text into Chunks| C[HuggingFace Embeddings]
    C -->|Converts Text to Numbers| D[(Chroma Vector DB)]
    E[User Query + Chat History] -->|Rewrites Query| F[Groq Llama 3.3]
    D -->|Finds Match| F
    F -->|Generates Answer| G[Streamlit Chat UI]


## 📁 Repository Structure
* `app_raw.py`: The optimized, production-ready version utilizing caching and structural modularity.
* `app_original.py`: My original standalone script where I first mapped out the baseline RAG logic.
