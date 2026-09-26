import pytest
from agents.supervisor import route_after_supervisor
from agents.research import should_continue_research
from agents.state import GraphState


class TestSupervisorRouting:
    """Test supervisor routing logic."""

    def test_route_retrieval(self):
        """Test that retrieval route is correctly identified."""
        state = GraphState(
            messages=[],
            session_id="test",
            route="retrieval",
            retrieved_docs=[],
            research_findings=[],
            research_query=None,
            research_iterations=0,
            max_research_iterations=3,
        )
        assert route_after_supervisor(state) == "retrieval"

    def test_route_research(self):
        """Test that research route is correctly identified."""
        state = GraphState(
            messages=[],
            session_id="test",
            route="research",
            retrieved_docs=[],
            research_findings=[],
            research_query=None,
            research_iterations=0,
            max_research_iterations=3,
        )
        assert route_after_supervisor(state) == "research"

    def test_route_response(self):
        """Test that response route is correctly identified."""
        state = GraphState(
            messages=[],
            session_id="test",
            route="response",
            retrieved_docs=[],
            research_findings=[],
            research_query=None,
            research_iterations=0,
            max_research_iterations=3,
        )
        assert route_after_supervisor(state) == "response"

    def test_route_default_on_none(self):
        """Test that None route defaults to response."""
        state = GraphState(
            messages=[],
            session_id="test",
            route=None,
            retrieved_docs=[],
            research_findings=[],
            research_query=None,
            research_iterations=0,
            max_research_iterations=3,
        )
        assert route_after_supervisor(state) == "response"

    def test_route_default_on_invalid(self):
        """Test that invalid route defaults to response."""
        state = GraphState(
            messages=[],
            session_id="test",
            route="invalid",  # type: ignore
            retrieved_docs=[],
            research_findings=[],
            research_query=None,
            research_iterations=0,
            max_research_iterations=3,
        )
        assert route_after_supervisor(state) == "response"


class TestResearchContinuation:
    """Test research iteration loop control."""

    def test_continue_research_below_max(self):
        """Test that research continues when below max iterations."""
        state = GraphState(
            messages=[],
            session_id="test",
            route=None,
            retrieved_docs=[],
            research_findings=[],
            research_query=None,
            research_iterations=1,
            max_research_iterations=3,
        )
        assert should_continue_research(state) == "research"

    def test_continue_research_at_zero(self):
        """Test that research continues from zero iterations."""
        state = GraphState(
            messages=[],
            session_id="test",
            route=None,
            retrieved_docs=[],
            research_findings=[],
            research_query=None,
            research_iterations=0,
            max_research_iterations=3,
        )
        assert should_continue_research(state) == "research"

    def test_stop_research_at_max(self):
        """Test that research stops at max iterations."""
        state = GraphState(
            messages=[],
            session_id="test",
            route=None,
            retrieved_docs=[],
            research_findings=[],
            research_query=None,
            research_iterations=3,
            max_research_iterations=3,
        )
        assert should_continue_research(state) == "response"

    def test_stop_research_above_max(self):
        """Test that research stops when exceeding max iterations."""
        state = GraphState(
            messages=[],
            session_id="test",
            route=None,
            retrieved_docs=[],
            research_findings=[],
            research_query=None,
            research_iterations=5,
            max_research_iterations=3,
        )
        assert should_continue_research(state) == "response"

    def test_research_with_max_one(self):
        """Test research iteration with max_iterations=1."""
        state = GraphState(
            messages=[],
            session_id="test",
            route=None,
            retrieved_docs=[],
            research_findings=[],
            research_query=None,
            research_iterations=0,
            max_research_iterations=1,
        )
        assert should_continue_research(state) == "research"

        state["research_iterations"] = 1
        assert should_continue_research(state) == "response"
