import streamlit as st
import requests
import os
import uuid
import json
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Chat", layout="wide")

st.title("🤖 Multi-Agent RAG Chat")
st.markdown("Chat with an AI-powered assistant that retrieves relevant information to answer your questions.")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "90"))

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "error_message" not in st.session_state:
    st.session_state.error_message = None

if "activity_log" not in st.session_state:
    st.session_state.activity_log = []

if "memory_info" not in st.session_state:
    st.session_state.memory_info = None


def refresh_memory_info():
    """Fetch the current session's stored memory from the backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/sessions/{st.session_state.session_id}/memory", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        st.session_state.memory_info = resp.json()
    except requests.exceptions.RequestException:
        # Non-fatal: memory panel just stays stale/empty if this fails.
        pass


def render_memory_panel():
    st.sidebar.markdown("### Session Memory")
    info = st.session_state.memory_info
    if not info:
        st.sidebar.caption("No memory yet.")
        return

    profile = info.get("user_profile") or {}
    if profile:
        st.sidebar.markdown("**About you:**")
        for key, value in profile.items():
            st.sidebar.markdown(f"- {key}: {value}")

    questions = info.get("previous_questions") or []
    if questions:
        st.sidebar.markdown(f"**Previous questions ({len(questions)}):**")
        for q in questions[-5:]:
            st.sidebar.markdown(f"- {q}")


ACTIVITY_ICONS = {
    "node": "🧭",
    "route": "🔀",
    "tool_call": "🔧",
    "retrieval_status": "📄",
    "memory_update": "💾",
    "validation": "✅",
    "final_response": "🏁",
}


def render_activity_panel(placeholder, events):
    if not events:
        placeholder.caption("No activity yet.")
        return
    lines = []
    for e in events:
        icon = ACTIVITY_ICONS.get(e["type"], "•")
        if e["type"] == "validation" and not e.get("passed", True):
            icon = "⚠️"
        lines.append(f"{icon} **{e.get('node', '')}** — {e['label']}")
    placeholder.markdown("\n\n".join(lines))

@st.dialog("⚠️ Error")
def show_error_dialog(error_text):
    st.markdown(error_text)
    if st.button("Close", key="error_close"):
        st.session_state.error_message = None
        st.rerun()

if st.session_state.error_message:
    show_error_dialog(st.session_state.error_message)

# Always-visible activity panel in sidebar
st.sidebar.markdown("### Agent Activity")
activity_placeholder = st.sidebar.empty()
render_activity_panel(activity_placeholder, st.session_state.activity_log)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        error_occurred = False

        # Reset activity log and re-render the panel (empty) when a new prompt is submitted
        st.session_state.activity_log = []
        render_activity_panel(activity_placeholder, st.session_state.activity_log)

        try:
            response = requests.post(
                f"{BACKEND_URL}/chat/stream",
                json={"message": prompt, "session_id": st.session_state.session_id},
                timeout=REQUEST_TIMEOUT,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                if event_type == "error":
                    st.session_state.error_message = event.get("detail", "Error processing request")
                    error_occurred = True
                    break

                elif event_type == "final":
                    full_response = event.get("content", "")
                    st.session_state.session_id = event.get("session_id", st.session_state.session_id)

                else:
                    # Add to activity log and update sidebar
                    st.session_state.activity_log.append(event)
                    render_activity_panel(activity_placeholder, st.session_state.activity_log)

        except requests.exceptions.ConnectionError:
            error_msg = f"Cannot connect to backend at {BACKEND_URL}. Please ensure the backend is running."
            st.session_state.error_message = error_msg
            error_occurred = True
        except requests.exceptions.Timeout:
            st.session_state.error_message = "Request timed out. Please try again."
            error_occurred = True
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            st.session_state.error_message = error_msg
            error_occurred = True
        except Exception as e:
            st.session_state.error_message = str(e)
            error_occurred = True

        if not error_occurred:
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            refresh_memory_info()
        else:
            st.session_state.messages.pop()
            st.rerun()

render_memory_panel()

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
st.sidebar.info(f"Backend URL: `{BACKEND_URL}`")

if st.sidebar.button("Clear Chat History"):
    try:
        requests.delete(f"{BACKEND_URL}/sessions/{st.session_state.session_id}", timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException:
        # Non-fatal: we still start a fresh session below even if the delete fails.
        pass
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.activity_log = []
    st.session_state.memory_info = None
    st.rerun()
