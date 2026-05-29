from app.services.retrieval_service import RetrievalService

_CONTEXT_HEADER = (
    "--- Background memory context (do not quote, do not reference as memory, "
    "weave naturally or ignore if irrelevant) ---"
)


class ReinjectionService:
    def __init__(self, retrieval_service: RetrievalService) -> None:
        self.retrieval_service = retrieval_service

    def build_injected_context(
        self,
        query: str,
        session_id: str | None = None,
        recent_turns: list[str] | None = None,
    ) -> tuple[str | None, list[dict]]:
        semantic_text, context_items = self.retrieval_service.retrieve_for_query(
            query,
            session_id=session_id,
            recent_turns=recent_turns,
        )
        if not semantic_text:
            return None, []
        return f"{_CONTEXT_HEADER}\n{semantic_text}", context_items
