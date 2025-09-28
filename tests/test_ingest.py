import os
import shutil
import tempfile
import pytest
from src.rag import ingest


@pytest.fixture
def temp_vectorstore_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_ingestion_creates_vectorstore(temp_vectorstore_dir):
    ingest.main(vectorstore_override=temp_vectorstore_dir)
    assert os.path.exists(temp_vectorstore_dir), "Vectorstore directory was not created"
    assert os.listdir(temp_vectorstore_dir), "Vectorstore directory is empty"


def test_ingestion_is_idempotent(temp_vectorstore_dir):
    ingest.main(vectorstore_override=temp_vectorstore_dir)
    first_files = set(os.listdir(temp_vectorstore_dir))
    ingest.main(vectorstore_override=temp_vectorstore_dir)
    second_files = set(os.listdir(temp_vectorstore_dir))
    assert first_files == second_files, "Vectorstore files changed unexpectedly"
