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
            full_response = f"⚠️ Error: Cannot connect to backend at {BACKEND_URL}. Please ensure the backend is running."
        except requests.exceptions.Timeout:
            full_response = "⚠️ Error: Request timed out. Please try again."
        except requests.exceptions.HTTPError as e:
            full_response = f"⚠️ Error: {e.response.status_code} - {e.response.text}"
        except Exception as e:
            full_response = f"⚠️ Error: {str(e)}"

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
st.sidebar.info(f"Backend URL: `{BACKEND_URL}`")

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
