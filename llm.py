"""
LLM dispatch layer — the only place in the codebase that talks to a model.

call_llm(system_prompt, user_prompt) -> str   # raw model text

Backend is selected by config.LLM_BACKEND ("anthropic" | "local").
"""

import config


def call_llm(system_prompt: str, user_prompt: str) -> str:
    if config.LLM_BACKEND == "anthropic":
        return _call_anthropic(system_prompt, user_prompt)
    elif config.LLM_BACKEND == "local":
        return _call_local(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {config.LLM_BACKEND!r}. Choose 'anthropic' or 'local'.")


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

def _call_anthropic(system_prompt: str, user_prompt: str) -> str:
    if not config.ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set. Export it before running.")

    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    message = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Local backend (Ollama) — TODO: implement when local model is ready
# ---------------------------------------------------------------------------

def _call_local(system_prompt: str, user_prompt: str) -> str:
    # TODO: replace with real Ollama call, e.g.:
    #
    #   import requests
    #   resp = requests.post(
    #       f"{config.OLLAMA_BASE_URL}/api/chat",
    #       json={
    #           "model": config.OLLAMA_MODEL,
    #           "messages": [
    #               {"role": "system", "content": system_prompt},
    #               {"role": "user",   "content": user_prompt},
    #           ],
    #           "stream": False,
    #       },
    #   )
    #   return resp.json()["message"]["content"]
    #
    raise NotImplementedError(
        "Local backend is not yet implemented. "
        f"Set LLM_BACKEND=anthropic or implement _call_local() in llm.py. "
        f"(Configured model: {config.OLLAMA_MODEL} at {config.OLLAMA_BASE_URL})"
    )
