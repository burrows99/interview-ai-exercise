"""LLM-as-judge evaluation for the RAG pipeline.

Three reference-free metrics (all 0-1, higher is better):
- faithfulness:       every claim in the answer is grounded in the retrieved context
- answer_relevancy:   the answer actually addresses the question
- context_relevancy:  the retrieved chunks are relevant to the question
"""

import logging

from ai_exercise.evaluation.rag_evaluator_prompts import RAGEvaluatorPrompts
from ai_exercise.llm.providers import LLMProvider

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """LLM-as-judge evaluator for RAG pipelines."""

    def __init__(self, llm: LLMProvider) -> None:
        """Initialise the evaluator with an LLM provider."""
        self._llm = llm
        self._prompts = RAGEvaluatorPrompts()

    # --- Utilities ---

    @staticmethod
    def _parse_score(raw: str) -> float:
        """Extract the first float found in a model response, clamped to [0, 1]."""
        for token in raw.replace(",", ".").split():
            try:
                value = float(token.strip("()."))
                return max(0.0, min(1.0, value))
            except ValueError:
                continue
        return 0.0

    # --- Scoring methods ---

    def score_faithfulness(
        self, query: str, context: list[str], answer: str
    ) -> float:
        """Score whether every claim in *answer* is supported by *context* (0-1)."""
        if not context:
            return 0.0
        prompt = self._prompts.faithfulness(query, "\n\n".join(context), answer)
        return self._parse_score(self._llm.get_completion(prompt))

    def score_answer_relevancy(self, query: str, answer: str) -> float:
        """Score whether *answer* actually addresses *query* (0-1)."""
        prompt = self._prompts.answer_relevancy(query, answer)
        return self._parse_score(self._llm.get_completion(prompt))

    def score_context_relevancy(self, query: str, context: list[str]) -> float:
        """Score what proportion of retrieved context is relevant to *query* (0-1)."""
        if not context:
            return 0.0
        prompt = self._prompts.context_relevancy(query, "\n\n".join(context))
        return self._parse_score(self._llm.get_completion(prompt))

    def evaluate(
        self, query: str, context: list[str], answer: str
    ) -> dict[str, float]:
        """Run all three metrics and return a scores dict."""
        scores = {
            "faithfulness": self.score_faithfulness(query, context, answer),
            "answer_relevancy": self.score_answer_relevancy(query, answer),
            "context_relevancy": self.score_context_relevancy(query, context),
        }
        logger.debug("Scores: %s", scores)
        return scores
