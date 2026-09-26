import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Chat", layout="wide")

st.title("🤖 Multi-Agent RAG Chat")
st.markdown("Chat with an AI-powered assistant that retrieves relevant information to answer your questions.")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "error_message" not in st.session_state:
    st.session_state.error_message = None

@st.dialog("⚠️ Error")
def show_error_dialog(error_text):
    st.markdown(error_text)
    if st.button("Close", key="error_close"):
        st.session_state.error_message = None
        st.rerun()

if st.session_state.error_message:
    show_error_dialog(st.session_state.error_message)

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

        try:
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={"message": prompt},
                timeout=30,
                stream=False
            )
            response.raise_for_status()

            data = response.json()
            full_response = data.get("response", "Sorry, I encountered an error processing your request.")

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
        else:
            st.session_state.messages.pop()
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
st.sidebar.info(f"Backend URL: `{BACKEND_URL}`")

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
