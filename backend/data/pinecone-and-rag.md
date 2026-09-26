# Pinecone and Retrieval-Augmented Generation (RAG)

## What is Retrieval-Augmented Generation?

RAG is a technique that combines information retrieval with text generation. Instead of relying solely on an LLM's training data, a RAG system:
1. Retrieves relevant documents from a knowledge base
2. Uses those documents as context for the LLM
3. Generates responses grounded in the retrieved information

This approach helps provide accurate, up-to-date information and reduces hallucinations.

## Vector Search and Embeddings

RAG systems rely on vector search:
- Text is converted to high-dimensional vectors called embeddings
- Similar text has similar embeddings
- A vector database finds semantically similar documents quickly

## Pinecone Overview

Pinecone is a managed vector database designed for production RAG applications:

### Key Features
- **Serverless Infrastructure**: No need to manage servers
- **High Performance**: Supports fast similarity search at scale
- **Built-in Replication**: Automatic failover and reliability
- **Metadata Filtering**: Filter results by metadata fields
- **Hybrid Search**: Combine vector search with keyword filtering

### Indexes in Pinecone
An index is a collection of vectors with associated metadata. Key properties:
- **Dimension**: Fixed size of embeddings (e.g., 1536 for OpenAI's text-embedding-3-small)
- **Metric**: Distance metric for similarity (e.g., cosine, euclidean, dotproduct)
- **Spec**: Environment specification (Serverless or Pod-based)

## How Pinecone Works in This Project

1. **Index Creation**: The system creates a serverless Pinecone index on first startup
2. **Document Ingestion**: Seed documents are split into chunks and embedded
3. **Embedding**: Each chunk is converted to a vector using the OpenRouter embeddings endpoint
4. **Storage**: Vectors and metadata are stored in the Pinecone index
5. **Retrieval**: When a user queries, their question is embedded and used to search for similar chunks
6. **Context Synthesis**: Retrieved chunks are provided to the LLM as context for answering

## Serverless Pinecone

Pinecone's serverless offering:
- **Automatic Scaling**: Resources scale automatically based on usage
- **Pay-as-You-Go**: Charged only for actual usage
- **Regional Deployment**: Choose your cloud region
- **No Operational Overhead**: Pinecone handles all infrastructure

## Vector Embeddings

Embeddings are numerical representations of text that capture semantic meaning:
- Shorter documents → shorter vectors
- Similar content → closer vectors in vector space
- Similarity search finds the nearest vectors to a query vector

### OpenRouter Embeddings
OpenRouter provides access to multiple embedding models through a unified API:
- Supports models like OpenAI's text-embedding-3-small and text-embedding-3-large
- Dimension varies by model (1536 for text-embedding-3-small)
- Same authentication as LLM calls (OPENROUTER_API_KEY)

## RAG Workflow in This System

1. **User Query**: User sends a message
2. **Supervisor Routes**: Determines if retrieval is needed
3. **Query Embedding**: User message is embedded
4. **Similarity Search**: Top-k similar chunks are retrieved from Pinecone
5. **Context Assembly**: Retrieved chunks are compiled as context
6. **Answer Generation**: LLM generates response using context
7. **Return Result**: User receives grounded, accurate answer

## Benefits of This Approach

1. **Accuracy**: Answers grounded in actual documents
2. **Freshness**: Can include recent documents
3. **Transparency**: Sources are traceable
4. **Scalability**: Vector search scales to millions of documents
5. **Efficiency**: Retrieval is fast (milliseconds)
