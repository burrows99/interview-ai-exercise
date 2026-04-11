"""Prompt construction helpers."""

# ruff: noqa: E501  — prompts contain intentional long prose lines


class RAGChatPrompts:
    """Static prompt builders for chat completions."""

    @staticmethod
    def answer(query: str, context: list[str]) -> str:
        """Build a prompt combining query and retrieved context."""
        if not context:
            return f"""You are a helpful assistant for StackOne API documentation.
You were asked the following question but no relevant documentation was found to answer it.
Politely tell the user you don't have enough information to answer their question and suggest they consult the StackOne documentation directly.

Question: {query}

Answer:"""

        context_str = "\n\n".join(context)
        return f"""You are a helpful assistant for StackOne API documentation.
Answer the question based solely on the provided context. If the context does not contain enough information to answer, say so clearly.

Context:
{context_str}

Question: {query}

Answer:"""
