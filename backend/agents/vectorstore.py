import os
import functools
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from .embeddings import OpenRouterEmbeddings, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_vectorstore() -> PineconeVectorStore:
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "multi-agent-rag-demo")
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")

    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")

    pc = Pinecone(api_key=api_key)

    # Create index if it doesn't exist
    existing_indexes = pc.list_indexes()
    index_names = [idx.name for idx in existing_indexes.indexes] if existing_indexes.indexes else []

    if index_name not in index_names:
        logger.info(f"Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        logger.info(f"Index '{index_name}' created successfully")
    else:
        logger.info(f"Index '{index_name}' already exists")

    index = pc.Index(index_name)
    embeddings = OpenRouterEmbeddings()

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text",
    )

    return vectorstore
