# Multi-Agent RAG Demo Project

## Overview

This is a multi-agent Retrieval-Augmented Generation (RAG) system built with LangGraph and FastAPI. The system combines a Streamlit frontend with a sophisticated backend that uses multiple specialized agents to answer user queries.

## Architecture

The system is built on a monorepo structure with two main components:

1. **Frontend**: A Streamlit chat interface that allows users to interact with the RAG system
2. **Backend**: A FastAPI REST API that implements the multi-agent orchestration

## Key Components

### Frontend (Streamlit)
- Provides a real-time chat interface
- Maintains conversation history client-side
- Communicates with the backend via HTTP requests
- Displays responses and error handling in a user-friendly manner

### Backend (FastAPI)
- Exposes a `/chat` endpoint that accepts user messages
- Implements a LangGraph-based multi-agent system
- Manages conversation context and routing
- Integrates with vector databases and LLMs

## Multi-Agent System

The backend uses four specialized agents working in coordination:

1. **Supervisor Agent**: Routes incoming user messages to appropriate handlers
2. **Retrieval Agent**: Performs vector search on document collections
3. **Research Agent**: Iteratively searches and synthesizes information
4. **Response Agent**: Generates final answers using context and findings

## Technology Stack

- **Frontend**: Streamlit 1.41.1
- **Backend**: FastAPI 0.141.1
- **Graph Orchestration**: LangGraph 1.2.11
- **Vector Store**: Pinecone
- **LLM Provider**: OpenRouter
- **Embeddings**: Custom OpenRouter implementation
- **Language**: Python 3.12+
