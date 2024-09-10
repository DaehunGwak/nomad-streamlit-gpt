import os

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def get_retriever_after_embedding(input_file, session_id, api_key):
    seperated_docs = _get_separated_docs_after_embedding(input_file, session_id)
    cached_embeddings = _get_cached_embeddings(input_file, session_id, api_key)

    vectorstore = FAISS.from_documents(embedding=cached_embeddings, documents=seperated_docs)
    return vectorstore.as_retriever()


def _get_separated_docs_after_embedding(input_file, session_id):
    file_content = input_file.read()
    file_path = f"./.cache/files/{session_id}"
    file_full_path = f"{file_path}/{input_file.name}"

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(file_full_path, "w") as target_file:
        target_file.write(file_content.decode("utf-8"))

    character_text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",
        chunk_size=600,
        chunk_overlap=100,
        separator="\n",
    )
    loader = TextLoader(file_full_path)
    return loader.load_and_split(text_splitter=character_text_splitter)


def _get_cached_embeddings(input_file, session_id, api_key):
    embeddings_cache_dir = f'./.cache/embeddings/{session_id}'

    if not os.path.exists(embeddings_cache_dir):
        os.makedirs(embeddings_cache_dir)

    openai_3_small_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
    file_store = LocalFileStore(f'{embeddings_cache_dir}/{input_file.name}')
    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=openai_3_small_embeddings,
        document_embedding_cache=file_store
    )
