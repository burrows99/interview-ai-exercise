"""Prompt builders for the RAG evaluator."""


class RAGEvaluatorPrompts:
    """Static prompt builders for the RAG evaluator."""

    @staticmethod
    def faithfulness(query: str, context_str: str, answer: str) -> str:
        """Prompt for scoring faithfulness."""
        return f"""You are a strict evaluation assistant.

Given a question, retrieved context, and an answer, score how faithfully the answer
is grounded in the context. A faithful answer only makes claims that are directly
supported by the context and does not introduce outside information.

Question: {query}

Context:
{context_str}

Answer: {answer}

Score faithfulness from 0.0 to 1.0 where:
  1.0 = every claim in the answer is explicitly supported by the context
  0.0 = the answer introduces facts not present in the context

Respond with a single float and nothing else."""

    @staticmethod
    def answer_relevancy(query: str, answer: str) -> str:
        """Prompt for scoring answer relevancy."""
        return f"""You are a strict evaluation assistant.

Given a question and an answer, score how well the answer addresses the question.

Question: {query}

Answer: {answer}

Score answer relevancy from 0.0 to 1.0 where:
  1.0 = the answer completely and directly addresses the question
  0.0 = the answer is entirely irrelevant to the question

Respond with a single float and nothing else."""

    @staticmethod
    def context_relevancy(query: str, context_str: str) -> str:
        """Prompt for scoring context relevancy."""
        return f"""You are a strict evaluation assistant.

Given a question and a set of retrieved context chunks, score how relevant the
retrieved context is to answering the question.

Question: {query}

Retrieved Context:
{context_str}

Score context relevancy from 0.0 to 1.0 where:
  1.0 = all retrieved chunks are highly relevant to answering the question
  0.0 = none of the retrieved chunks are relevant to the question

Respond with a single float and nothing else."""
