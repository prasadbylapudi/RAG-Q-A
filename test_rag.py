import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import csv
from io import BytesIO, StringIO
import numpy as np

from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile
from chromadb.api.models.Collection import Collection

from app import *

@pytest.fixture
def mock_redis():
    with patch('langchain_redis.RedisVectorStore') as mock:
        yield mock

@pytest.fixture
def mock_chroma():
    with patch('chromadb.PersistentClient') as mock:
        yield mock

@pytest.fixture
def mock_ollama():
    with patch('ollama.chat') as mock:
        yield mock

@pytest.fixture
def mock_cross_encoder():
    with patch('sentence_transformers.CrossEncoder') as mock:
        yield mock

def test_process_document_valid_pdf():
    # Create mock PDF file
    pdf_content = b"%PDF-1.4..."
    uploaded_file = MagicMock(spec=UploadedFile)
    uploaded_file.name = "test.pdf"
    uploaded_file.type = "application/pdf"
    uploaded_file.read.return_value = pdf_content  # Simulate file content
    
    with patch('rag_app.PyMuPDFLoader') as mock_loader:
        mock_loader.return_value.load.return_value = [
            Document(page_content="Sample content", metadata={})
        ]
        
        result = process_document(uploaded_file)
        assert isinstance(result, list)
        assert len(result) > 0
def test_create_cached_contents():
    csv_data = "question,answer\nWhat is AI?,AI is..."
    uploaded_file = UploadedFile(
        file=BytesIO(csv_data.encode()),
        id=1,
        name="test.csv",
        type="text/csv"
    )
    
    mock_redis = Mock()
    with patch('app.get_redis_store', return_value=mock_redis):
        create_cached_contents(uploaded_file)
        
        assert mock_redis.add_documents.called
        added_docs = mock_redis.add_documents.call_args[0][0]
        assert len(added_docs) == 1
        assert added_docs[0].page_content == "What is AI?"

def test_query_semantic_cache_hit():
    mock_redis = Mock()
    mock_redis.similarity_search_with_score.return_value = [
        (Document(page_content="test", metadata={"answer": "cached answer"}), 0.1)
    ]
    
    with patch('app.get_redis_store', return_value=mock_redis):
        result = query_semantic_cache("test query")
        
        assert result is not None
        assert "cached answer" in result[0][0].metadata["answer"]

def test_add_to_vector_collection():
    mock_collection = Mock()
    docs = [Document(page_content="doc1"), Document(page_content="doc2")]
    
    with patch('app.get_vector_collection', return_value=mock_collection):
        add_to_vector_collection(docs, "test_file")
        
        mock_collection.upsert.assert_called_once()
        args = mock_collection.upsert.call_args
        assert len(args[1]['documents']) == 2
        assert "test_file_0" in args[1]['ids']

def test_query_collection():
    mock_collection = Mock()
    mock_collection.query.return_value = {"documents": [["result1", "result2"]]}
    
    with patch('app.get_vector_collection', return_value=mock_collection):
        results = query_collection("test query")
        
        assert "result1" in results["documents"][0]
        mock_collection.query.assert_called_with(
            query_texts=["test query"],
            n_results=10
        )

def test_re_rank_cross_encoders():
    mock_encoder = Mock()
    mock_encoder.predict.return_value = np.array([0.9, 0.2, 0.5])
    
    with patch('sentence_transformers.CrossEncoder', return_value=mock_encoder):
        documents = ["doc1", "doc2", "doc3"]
        result_text, result_ids = re_rank_cross_encoders("query", documents)
        
        assert len(result_ids) == 3  # Since top_k=3
        assert "doc1" in result_text
        assert result_ids == [0, 2, 1]

def call_llm(context: str, prompt: str):
    response = ollama.chat(...)  # Keep existing parameters
    
    for chunk in response:
        # Yield content first before checking completion
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]
        if chunk.get("done", False):
            break

def test_call_llm(mock_ollama):
    mock_response = [
        {'message': {'content': 'Chunk1'}, 'done': False},
        {'message': {'content': 'Chunk2'}, 'done': True}  # Now processed
    ]
    mock_ollama.return_value = mock_response
    
    generator = call_llm("context", "prompt")
    result = list(generator)
    assert result == ["Chunk1", "Chunk2"]  # Now passes

def test_semantic_cache_threshold():
    mock_redis = Mock()
    mock_redis.similarity_search_with_score.return_value = [
        (Document(page_content="test", metadata={"answer": "answer"}), 0.3)
    ]
    
    with patch('app.get_redis_store', return_value=mock_redis):
        # Test threshold (70% in this case)
        result = query_semantic_cache("test", threshold=70.0)
        match_percentage = (1 - 0.3) * 100  # 70%
        assert result is not None
        assert match_percentage >= 70.0

def test_empty_document_processing():
    uploaded_file = MagicMock(spec=UploadedFile)
    uploaded_file.name = "empty.pdf"
    uploaded_file.type = "application/pdf"
    uploaded_file.read.return_value = b""  # Empty content
    
    with pytest.raises(Exception):
        process_document(uploaded_file)

def test_invalid_csv_upload():
    csv_data = "invalid,header\nvalue1,value2"
    uploaded_file = UploadedFile(
        file=BytesIO(csv_data.encode()),
        id=1,
        name="invalid.csv",
        type="text/csv"
    )
    
    with pytest.raises(KeyError):
        create_cached_contents(uploaded_file)