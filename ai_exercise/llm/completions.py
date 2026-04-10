"""Prompt construction helpers."""


def create_prompt(query: str, context: list[str]) -> str:
    """Create a prompt combining query and context."""
    context_str = "\n\n".join(context)
    return f"""Please answer the question based on the following context:

Context:
{context_str}

Question: {query}

Answer:"""
