import datetime


def _now() -> str:
    return datetime.datetime.utcnow().isoformat()


def build_activity_events(node: str, payload: dict) -> list[dict]:
    """Translate a LangGraph node's state update into activity-panel events."""
    events: list[dict] = []

    if node == "supervisor":
        route = payload.get("route")
        events.append({"type": "node", "node": node, "label": "Supervisor analyzed the message", "timestamp": _now()})
        events.append({"type": "route", "node": node, "route": route, "label": f"Routed to: {route}", "timestamp": _now()})

    elif node == "retrieval":
        docs = payload.get("retrieved_docs", [])
        sources = [d.get("source", "unknown") for d in docs]
        events.append({"type": "tool_call", "node": node, "tool": "vectorstore.asimilarity_search", "label": "Searching document index", "timestamp": _now()})
        events.append({"type": "retrieval_status", "node": node, "count": len(docs), "sources": sources, "label": f"Retrieved {len(docs)} chunk(s)", "timestamp": _now()})
        events.append({
            "type": "validation", "node": node, "passed": len(docs) > 0,
            "label": "Relevant context found" if docs else "No relevant documents found",
            "timestamp": _now(),
        })

    elif node == "research":
        iterations = payload.get("research_iterations")
        query = payload.get("research_query")
        findings = payload.get("research_findings", [])
        events.append({"type": "tool_call", "node": node, "tool": "vectorstore.asimilarity_search", "label": f"Research iteration {iterations}: '{query}'", "timestamp": _now()})
        events.append({"type": "memory_update", "node": node, "fields": {"research_iterations": iterations, "findings_count": len(findings)}, "label": f"Findings so far: {len(findings)}", "timestamp": _now()})
        events.append({
            "type": "validation", "node": node, "passed": len(findings) > 0,
            "label": "Research found supporting findings" if findings else "No findings gathered yet",
            "timestamp": _now(),
        })

    elif node == "response":
        events.append({"type": "final_response", "node": node, "label": "Generating final response", "timestamp": _now()})

    return events
