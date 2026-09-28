import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.graph import build_graph
from agents.memory import (
    merge_user_profile,
    parse_memory_output,
    render_memory_block,
    render_previous_questions,
    select_relevant_interactions,
)
from agents.state import Interaction


def _base_state(message: str, session_id: str = "test-session") -> dict:
    return {
        "messages": [HumanMessage(content=message)],
        "session_id": session_id,
        "route": None,
        "retrieved_docs": [],
        "research_findings": [],
        "research_query": None,
        "research_iterations": 0,
        "max_research_iterations": 3,
    }


class TestParseMemoryOutput:
    def test_parses_valid_json(self):
        raw = '{"standalone_question": "What is Pinecone?", "user_facts": {"name": "Sam"}}'
        question, facts = parse_memory_output(raw, fallback_question="fallback")
        assert question == "What is Pinecone?"
        assert facts == {"name": "Sam"}

    def test_falls_back_on_invalid_json(self):
        question, facts = parse_memory_output("not json at all", fallback_question="original question")
        assert question == "original question"
        assert facts == {}

    def test_strips_markdown_fences(self):
        raw = '```json\n{"standalone_question": "Q?", "user_facts": {}}\n```'
        question, facts = parse_memory_output(raw, fallback_question="fallback")
        assert question == "Q?"

    def test_missing_standalone_question_falls_back(self):
        raw = '{"user_facts": {"team": "platform"}}'
        question, facts = parse_memory_output(raw, fallback_question="fallback")
        assert question == "fallback"
        assert facts == {"team": "platform"}


class TestMergeUserProfile:
    def test_merges_new_facts(self):
        merged = merge_user_profile({"name": "Sam"}, {"team": "platform"})
        assert merged == {"name": "Sam", "team": "platform"}

    def test_new_facts_override_existing(self):
        merged = merge_user_profile({"name": "Sam"}, {"name": "Samantha"})
        assert merged["name"] == "Samantha"

    def test_drops_sensitive_keys(self):
        merged = merge_user_profile({}, {"api_key": "sk-123", "name": "Sam"})
        assert "api_key" not in merged
        assert merged["name"] == "Sam"

    def test_caps_total_facts(self):
        existing = {f"fact_{i}": str(i) for i in range(20)}
        merged = merge_user_profile(existing, {"fact_new": "value"})
        assert len(merged) == 20
        assert "fact_new" in merged
        assert "fact_0" not in merged  # oldest fact evicted


class TestSelectRelevantInteractions:
    def _interaction(self, turn: int, question: str, answer: str = "an answer") -> Interaction:
        return {
            "turn": turn,
            "question": question,
            "standalone_question": question,
            "answer": answer,
            "route": "retrieval",
        }

    def test_empty_interactions_returns_empty(self):
        assert select_relevant_interactions([], "anything") == []

    def test_includes_keyword_matches(self):
        interactions = [
            self._interaction(1, "What is Pinecone used for?", "Pinecone is a vector database."),
            self._interaction(2, "How do I start the frontend?", "Run streamlit run app.py."),
        ]
        result = select_relevant_interactions(interactions, "Tell me more about Pinecone", k=3)
        assert any(i["turn"] == 1 for i in result)

    def test_always_includes_last_two_turns(self):
        interactions = [self._interaction(i, f"question {i}") for i in range(1, 6)]
        result = select_relevant_interactions(interactions, "unrelated query xyz", k=3)
        turns = {i["turn"] for i in result}
        assert {4, 5}.issubset(turns)

    def test_result_is_chronologically_ordered(self):
        interactions = [
            self._interaction(1, "pinecone question", "pinecone answer"),
            self._interaction(2, "unrelated"),
            self._interaction(3, "unrelated"),
        ]
        result = select_relevant_interactions(interactions, "pinecone", k=3)
        turn_numbers = [i["turn"] for i in result]
        assert turn_numbers == sorted(turn_numbers)


class TestRenderHelpers:
    def test_render_memory_block_empty(self):
        assert render_memory_block({}, None) == ""

    def test_render_memory_block_includes_profile(self):
        block = render_memory_block({"name": "Sam"}, None)
        assert "Sam" in block

    def test_render_previous_questions_empty(self):
        assert render_previous_questions([]) == ""

    def test_render_previous_questions_lists_all(self):
        interactions = [
            {"turn": 1, "question": "q1", "standalone_question": "q1", "answer": "a1", "route": "response"},
            {"turn": 2, "question": "q2", "standalone_question": "q2", "answer": "a2", "route": "response"},
        ]
        rendered = render_previous_questions(interactions)
        assert "q1" in rendered and "q2" in rendered


@pytest.mark.asyncio
class TestMemoryAcrossTurns:
    async def test_user_profile_persists_across_turns(self, mocked_graph_deps):
        graph = build_graph()
        thread = {"configurable": {"thread_id": "memory-test-1"}}

        await graph.ainvoke(_base_state("Hi, I'm Sam and I'm on the platform team."), config=thread)
        state = await graph.aget_state(config=thread)

        assert state.values.get("user_profile", {}).get("name") == "Sam"

    async def test_previous_questions_are_tracked(self, mocked_graph_deps):
        graph = build_graph()
        thread = {"configurable": {"thread_id": "memory-test-2"}}

        await graph.ainvoke(_base_state("How do I start the backend?"), config=thread)
        await graph.ainvoke(_base_state("What about the frontend?"), config=thread)

        state = await graph.aget_state(config=thread)
        interactions = state.values.get("interactions", [])
        assert len(interactions) == 2
        assert state.values.get("turn_count") == 2

    async def test_sessions_do_not_share_memory(self, mocked_graph_deps):
        graph = build_graph()

        await graph.ainvoke(
            _base_state("Hi, I'm Alex."), config={"configurable": {"thread_id": "session-a"}}
        )
        state_b = await graph.aget_state(config={"configurable": {"thread_id": "session-b"}})

        assert state_b.values.get("user_profile", {}) == {}

    async def test_initial_state_does_not_wipe_existing_memory(self, mocked_graph_deps):
        """Regression test: _build_initial_state-style input must not include
        user_profile/interactions keys, since LangGraph merges partial updates
        and an empty dict there would erase memory on every turn."""
        graph = build_graph()
        thread = {"configurable": {"thread_id": "memory-test-3"}}

        await graph.ainvoke(_base_state("Hi, I'm Jordan."), config=thread)
        # Second turn's initial state (as main.py builds it) has no user_profile key at all.
        second_state = _base_state("What's my name?")
        assert "user_profile" not in second_state
        assert "interactions" not in second_state

        await graph.ainvoke(second_state, config=thread)
        state = await graph.aget_state(config=thread)
        assert state.values.get("user_profile", {}).get("name") == "Jordan"
