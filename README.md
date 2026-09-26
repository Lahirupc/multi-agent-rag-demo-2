# Multi-Agent RAG Demo

A monorepo containing a frontend and backend for a multi-agent Retrieval-Augmented Generation (RAG) chat application.

## Project Structure

This project is organized as a monorepo with two main parts:

### Frontend
- **Framework**: Streamlit
- **Description**: A simple chat interface for streaming input and output
- **Features**: Real-time message streaming for user interactions

### Backend
- **Framework**: FastAPI
- **Package Manager**: uv
- **Description**: REST API that processes chat requests using LangChain and LLMs
- **Key Endpoint**:
  - `POST /chat` - Processes chat requests by passing them to an LLM via LangChain and returns the response

## Architecture

The frontend communicates with the backend's `/chat` endpoint to process user messages through an LLM, enabling a seamless chat experience with streaming capabilities.

## Getting Started

Refer to the respective directories for setup and installation instructions for the frontend and backend components.