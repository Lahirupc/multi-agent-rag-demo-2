import os
import functools
from langchain_openrouter import ChatOpenRouter


@functools.lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenRouter:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_id = os.getenv("CHAT_MODEL", "liquid/lfm-2.5-2.6b:free")

    return ChatOpenRouter(
        model=model_id,
        api_key=api_key,
        temperature=0.7,
    )
