from app.llm.client import LLMClient
from app.pipeline.gist.gist_validator import GistValidator


class GistGenerator:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        self.validator = GistValidator()

    def generate_batch(self, segments: list[list[dict]]) -> list[dict | None]:
        results: list[dict | None] = []
        for segment in segments:
            try:
                gist_data = self.llm_client.build_gist(segment)
                results.append(gist_data if self.validator.is_valid(gist_data) else None)
            except Exception:
                results.append(None)
        return results
