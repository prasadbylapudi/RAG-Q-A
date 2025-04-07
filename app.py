import csv
import os
import sys
import tempfile
from io import StringIO
import numpy as np
import chromadb
import ollama
import streamlit as st
# from chromadb.utils.embedding_functions.ollama_embedding_function import (
#     OllamaEmbeddingFunction,
# )
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


def get_redis_store() -> RedisVectorStore:
    """Gets or creates a Redis vector store for caching embeddings.

    Creates an Ollama embeddings object using the nomic-embed-text model and initializes
    a Redis vector store with cosine similarity metric for storing cached question-answer pairs.
    """
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
    )
    # st.header("Embeddings")
    # st.write(embeddings)
    return RedisVectorStore(
        embeddings,
        config=RedisConfig(
            index_name="cached_contents",
            redis_url="redis://localhost:6379",
            distance_metric="COSINE",
            metadata_schema=[
                {"name": "answer", "type": "text"},
            ],
        ),
    )


def create_cached_contents(uploaded_file: UploadedFile) -> list[Document]:
    """Creates cached question-answer pairs from an uploaded CSV file.
    Takes an uploaded CSV file containing question-answer pairs, converts them to Document
    objects and adds them to a Redis vector store for caching.

    takes arguments  uploaded_file: A Streamlit UploadedFile object containing the CSV data with
            'question' and 'answer' columns.

    """
    data = uploaded_file.getvalue().decode("utf-8")
    csv_reader = csv.DictReader(StringIO(data))

    docs = []
    for row in csv_reader:
        docs.append(
            Document(page_content=row["question"], metadata={"answer": row["answer"]})
        )
    vector_store = get_redis_store()
    vector_store.add_documents(docs)
    st.success("Cache contents added!")


def query_semantic_cache(query: str, n_results: int = 1, threshold: float = 80.0):
    """Queries the semantic cache for similar questions and returns cached results if found.

    Args:
        query: The search query text to find relevant cached results.
        n_results: Maximum number of results to return. Defaults to 1.
        threshold: Minimum similarity score threshold (0-100) for returning cached results.
            Defaults to 80.0.
    """
    vector_store = get_redis_store()
    results = vector_store.similarity_search_with_score(query, k=n_results)

    if not results:
        return None

    match_percentage = (1 - abs(results[0][1])) * 100
    if match_percentage >= threshold:
        return results
    return None


    # Store uploaded file as a temp file
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)  # Delete temp file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    # Store uploaded file as a temp file
    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()

    # Ensure the file is closed before deleting
    os.unlink(temp_file_path)  # Delete temp file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)


def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage.

    Creates an Ollama embedding function using the nomic-embed-text model and initializes
    a persistent ChromaDB client. Returns a collection that can be used to store and
    query document embeddings.

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
            function and cosine similarity space.

            cosine_distance = 1 - cosine_similarity = 1-0.98 = 0.02 (very close)
            cosine similarity calculates the cosine of the angle between two vectors, how similar two vectors are.
                        A = "I love pizza"
                        B = "Pizza is great"  - > converted into vectors.
                            A = [0.1, 0.8, 0.3]
                            B = [0.2, 0.7, 0.4]
           cosine_similarity = dot(A, B) / (||A|| * ||B||) = 0.98 (very similar)
            dot(A, B) = (0.1*0.2) + (0.8*0.7) + (0.3*0.4)
                      = 0.02 + 0.56 + 0.12
                      = 0.70
            ||A|| = sqrt(0.1² + 0.8² + 0.3²) = sqrt(0.01 + 0.64 + 0.09) = sqrt(0.74) ≈ 0.86
            ||B|| = sqrt(0.2² + 0.7² + 0.4²) = sqrt(0.04 + 0.49 + 0.16) = sqrt(0.69) ≈ 0.83


            Meaning	            Cosine Distance	     Cosine Similarity
            Very similar	     0.0 – 0.3	        0.7 – 1.0
            Somewhat similar	~0.4 – 0.6	        ~0.6 – 0.4
            Not similar at all	 ~0.8 – 1.0	        0.2 – 0.0


            Euclidian distance - only cares about how big vectors are 
            consine similarity - cares about the angle between vectors
    """
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./qa-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store!")


def query_collection(prompt: str, n_results: int = 10):
    """Queries the vector collection with a given prompt to retrieve relevant documents.
    Query results containing documents, distances and metadata from the collection.

    """
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


def call_llm(context: str, prompt: str):
    """Calls the language model with context and prompt to generate a response.

    Uses Ollama to stream responses from a language model by providing context and a
    question prompt. The model uses a system prompt to format and ground its responses appropriately.
    """
    response = ollama.chat(
        model="llama3.2:3b",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


# def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
#     """Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

#     Uses the MS MARCO MiniLM cross-encoder model to re-rank the input documents based on
#     their relevance to the query prompt. Returns the concatenated text of the top 3 most
#     relevant documents along with their indices.

#     """
#     relevant_text = ""
#     relevant_text_ids = []

#     encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#     ranks = encoder_model.rank(prompt, documents, top_k=3)
#     for rank in ranks:
#         relevant_text += documents[rank["corpus_id"]]
#         relevant_text_ids.append(rank["corpus_id"])

#     return relevant_text, relevant_text_ids



def re_rank_cross_encoders(prompt: str, documents: list[str]) -> tuple[str, list[int]]:
    """Re-ranks documents using a cross-encoder for semantic similarity to the prompt.
        Cross-Encoder is a transformer model (like BERT or RoBERTa) that takes both the query and document together as input.
        It then generates a score for each document based on how similar the query and document are.

    """
    #this instance have context window of around 500.
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Prepare pairs of (prompt, document) for scoring
    pairs = [[prompt, doc] for doc in documents]
    scores = encoder_model.predict(pairs)

    # Get indices of top 3 documents
    top_indices = np.argsort(scores)[::-1][:3]
    
    # Combine top-ranked documents into one string (or return the best one)
    top_documents = [documents[i] for i in top_indices]
    # st.header(" top_documents")
    # st.write(top_documents)
    return "\n\n".join(top_documents), top_indices.tolist()


if __name__ == "__main__":
    # Document Upload Area
    with st.sidebar:
        st.set_page_config(page_title="RAG Questiona & Answer")
        uploaded_file = st.file_uploader(
            "** Upload PDF files for QnA**",
            type=["pdf", "csv"],
            accept_multiple_files=False,
            help="Upload csv for cached results only",
        )
        upload_option = st.radio(
            "Upload options:",
            options=["Primary", "Cache"],
            help="Choose Primary for uploading document for QnA.\n\nChoose Cache for uploading cached results",
        )

        if (
            uploaded_file
            and upload_option == "Primary"
            and uploaded_file.name.split(".")[-1] == "csv"
        ):
            st.error("CSV is only allowed for 'Cache' option.")
            sys.exit(1)

        process = st.button(
            " Process",
        )
        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )

            if upload_option == "Cache":
                all_splits = create_cached_contents(uploaded_file)
            else:
                all_splits = process_document(uploaded_file)
                # st.write(all_splits)
                add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    # Question and Answer Area
    st.header("RAG Question Answer")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button(
        "🔥 Ask",
    )

    if ask and prompt:
        cached_results = query_semantic_cache(query=prompt)

        if cached_results:  
            st.write(cached_results[0][0].metadata["answer"].replace("\\n", "\n"))
        else:
            results = query_collection(prompt=prompt)

            context = results.get("documents")[0]
            if not context:
                st.write("No results found.")
                sys.exit(1)
            relevant_text, relevant_text_ids = re_rank_cross_encoders(prompt,context)
            response = call_llm(context=relevant_text, prompt=prompt)
            st.write_stream(response)

            with st.expander("See retrieved documents"):
                st.write(results)

            with st.expander("See most relevant document ids"):
                st.write(relevant_text_ids)
                st.write(relevant_text)
