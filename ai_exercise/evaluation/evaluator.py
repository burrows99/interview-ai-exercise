"""LLM-as-judge evaluation for the RAG pipeline.

Three reference-free metrics (all 0-1, higher is better):
- faithfulness:       every claim in the answer is grounded in the retrieved context
- answer_relevancy:   the answer actually addresses the question
- context_relevancy:  the retrieved chunks are relevant to the question
"""

from ai_exercise.llm.provider import LLMProvider


def _parse_score(raw: str) -> float:
    """Extract the first float found in a model response, clamped to [0, 1]."""
    for token in raw.replace(",", ".").split():
        try:
            value = float(token.strip("()."))
            return max(0.0, min(1.0, value))
        except ValueError:
            continue
    return 0.0


def score_faithfulness(
    llm: LLMProvider, query: str, context: list[str], answer: str
) -> float:
    """Score whether every claim in *answer* is supported by *context* (0-1)."""
    if not context:
        # No context was retrieved; faithfulness is undefined — return 0
        return 0.0

    context_str = "\n\n".join(context)
    prompt = f"""You are a strict evaluation assistant.

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

    raw = llm.get_completion(prompt)
    return _parse_score(raw)


def score_answer_relevancy(llm: LLMProvider, query: str, answer: str) -> float:
    """Score whether *answer* actually addresses *query* (0-1)."""
    prompt = f"""You are a strict evaluation assistant.

Given a question and an answer, score how well the answer addresses the question.

Question: {query}

Answer: {answer}

Score answer relevancy from 0.0 to 1.0 where:
  1.0 = the answer completely and directly addresses the question
  0.0 = the answer is entirely irrelevant to the question

Respond with a single float and nothing else."""

    raw = llm.get_completion(prompt)
    return _parse_score(raw)


def score_context_relevancy(llm: LLMProvider, query: str, context: list[str]) -> float:
    """Score what proportion of the retrieved context is relevant to *query* (0-1)."""
    if not context:
        return 0.0

    context_str = "\n\n".join(context)
    prompt = f"""You are a strict evaluation assistant.

Given a question and a set of retrieved context chunks, score how relevant the
retrieved context is to answering the question.

Question: {query}

Retrieved Context:
{context_str}

Score context relevancy from 0.0 to 1.0 where:
  1.0 = all retrieved chunks are highly relevant to answering the question
  0.0 = none of the retrieved chunks are relevant to the question

Respond with a single float and nothing else."""

    raw = llm.get_completion(prompt)
    return _parse_score(raw)


def evaluate(
    llm: LLMProvider,
    query: str,
    context: list[str],
    answer: str,
) -> dict[str, float]:
    """Run all three metrics and return a scores dict."""
    return {
        "faithfulness": score_faithfulness(llm, query, context, answer),
        "answer_relevancy": score_answer_relevancy(llm, query, answer),
        "context_relevancy": score_context_relevancy(llm, query, context),
    }
