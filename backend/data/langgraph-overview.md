# LangGraph Overview

## What is LangGraph?

LangGraph is a library for building stateful, multi-agent applications using large language models. It enables the creation of complex workflows where multiple agents can coordinate with each other to solve problems.

## Key Concepts

### State Graph
A directed graph where nodes represent agents or processing steps, and edges represent the flow of information. Each node receives the current state and can modify it before passing it to the next node.

### StateGraph
A specific type of graph where:
- Each node is an async function that takes the current state and returns updates
- State is accumulated using reducer functions (like `add_messages` for conversation history)
- Conditional edges allow branching based on the current state

### Checkpointing
LangGraph supports checkpointing to save and resume graph execution. This enables:
- Multi-turn conversations with persistent memory
- Recovery from interruptions
- Thread-based conversation isolation

### MemorySaver
An in-memory checkpoint storage implementation provided by LangGraph. It stores state snapshots keyed by thread ID, allowing different conversation threads to maintain independent state.

## How It Works in This Project

1. **Initialization**: The graph is built once at application startup with a MemorySaver checkpointer
2. **Invocation**: When a user sends a message, the graph is invoked with the new message and a thread ID (session ID)
3. **State Management**: The checkpointer retrieves prior state for that thread, and the new message is accumulated with the add_messages reducer
4. **Agent Coordination**: Nodes (agents) are executed in sequence based on the conditional edges
5. **Persistence**: After execution, the updated state is stored in the checkpointer for future turns

## Nodes vs Agents

In LangGraph:
- A **node** is a Python async function that processes the current state
- An **agent** typically refers to a node that uses an LLM to make decisions

## Common Patterns

### Conditional Edges
Allow the graph to branch based on state content. For example:
- Supervisor routes to Retrieval, Research, or Response based on its decision
- Research continues looping or moves to Response based on iteration count

### Loops
Nodes can route back to themselves (like Research → Research) to create iterative processes.

### Sequential Flow
Multiple nodes can be connected in sequence (like Retrieval → Response) where one always follows another.

## Advantages for Multi-Agent Systems

1. **Explicit Control Flow**: Clear definition of how agents interact
2. **State Management**: Built-in handling of shared context
3. **Async Support**: Fully asynchronous for performance
4. **Checkpointing**: Native support for multi-turn memory
5. **Debugging**: Graph structure makes agent interactions transparent
