# Frequently Asked Questions

## General Questions

### What is this system?
This is a multi-agent RAG (Retrieval-Augmented Generation) system that combines:
- A Streamlit chat interface for user interactions
- A FastAPI backend for processing
- Multiple specialized AI agents that work together
- A Pinecone vector database for document search
- OpenRouter LLM integration

### How do the agents work together?
The system uses four agents in a coordinated workflow:
1. **Supervisor**: Analyzes the user's question and decides what kind of response is needed
2. **Retrieval**: Searches the document collection for relevant information
3. **Research**: Performs deeper investigation by iteratively refining searches
4. **Response**: Generates the final answer using retrieved information as context

### What does "multi-turn conversation" mean?
Multi-turn means the system remembers previous messages in a conversation. Each conversation has a unique session ID, and the LangGraph checkpointer stores the conversation history for that session.

## Technical Questions

### What embedding model is used?
The system uses OpenAI's text-embedding-3-small (1536 dimensions) accessed through OpenRouter's API.

### How are documents stored?
Documents are stored in Pinecone, a managed vector database. Documents are:
1. Split into chunks (1000 tokens with 200-token overlap)
2. Converted to embeddings (1536-dimensional vectors)
3. Stored with metadata (source filename, chunk index)

### What happens when I ask a question?
1. Your message goes to the Supervisor agent
2. Supervisor determines if retrieval, research, or simple response is needed
3. If retrieval is needed, documents are searched using vector similarity
4. If research is needed, the Research agent iteratively searches for deeper insights
5. The Response agent generates an answer using retrieved documents as context
6. Your message and the answer are stored in the conversation history

### How long do conversations last?
Conversation history persists for as long as you maintain the same session ID. When you refresh the page or clear chat history, a new session ID is generated and conversation history starts fresh.

## Usage Questions

### How do I clear the conversation?
Click the "Clear Chat History" button in the sidebar. This creates a new session ID, so the backend thread will no longer remember previous messages.

### What if I get an error?
Check the following:
1. Ensure the backend is running (`python main.py` from the backend directory)
2. Ensure all environment variables are set (.env file exists with API keys)
3. Ensure Pinecone credentials are valid
4. Ensure internet connection for OpenRouter API calls

### Can I use different LLM models?
Yes, you can change the model in the backend/.env file:
- Set `CHAT_MODEL` to any OpenRouter-supported model
- Set `EMBEDDING_MODEL` to any OpenRouter-supported embedding model

### How many research iterations happen?
By default, the Research agent will perform up to 3 iterations of refinement. This is configurable via the `MAX_RESEARCH_ITERATIONS` environment variable.

## Customization

### How do I add my own documents?
1. Add .md or .txt files to the `backend/data/` directory
2. Restart the backend server
3. The documents will be automatically ingested into Pinecone

### How do I change the vector database?
The system uses Pinecone's serverless offering. Alternative vector databases (Chroma, FAISS, Weaviate) could be substituted by:
1. Installing the corresponding LangChain integration package
2. Modifying `agents/vectorstore.py` to use the alternative database
3. Updating the index creation and retrieval logic

### Can I modify the agent behavior?
Yes, each agent is in its own module:
- `agents/supervisor.py`: Modify routing logic
- `agents/retrieval.py`: Modify search parameters (k=4 results)
- `agents/research.py`: Modify iteration logic
- `agents/response.py`: Modify answer generation
