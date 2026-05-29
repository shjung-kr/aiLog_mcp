from dataclasses import dataclass

from app.llm.client import LLMClient

SEMANTIC_TEXT_METADATA_KEY = "semantic_text"
EMBEDDING_SOURCE_VERSION_METADATA_KEY = "embedding_source_version"
EMBEDDING_SOURCE_VERSION = "episode_semantic_evidence_v1"
SEMANTIC_DETAIL_KEYS = (
    "user_goal",
    "context",
    "decision_or_insight",
    "emotional_or_situational_cue",
    "representative_snippets",
)


@dataclass(frozen=True)
class BuiltEpisode:
    title: str
    summary: str
    episode_type: str
    rawlog_ids: list[str]
    emotion_signal: str | None
    importance_score: float | None
    keywords: list[str] | None
    semantic_text: str
    metadata: dict


class EpisodeBuilder:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def build_from_gists(self, gist_segments: list[dict], valid_rawlog_ids: set[str]) -> list[BuiltEpisode]:
        if not gist_segments:
            return []

        episodes: list[BuiltEpisode] = []
        for item in self.llm_client.build_episodes_from_gists(gist_segments):
            rawlog_ids = [rid for rid in item.get("rawlog_ids", []) if rid in valid_rawlog_ids]
            if not rawlog_ids:
                continue

            title = str(item.get("title") or "Untitled episode").strip()
            summary = str(item.get("summary") or "Semantic episode generated from gists.").strip()
            episode_type = str(item.get("episode_type") or "topic").strip()
            keywords = item.get("keywords") if isinstance(item.get("keywords"), list) else None
            semantic_text = self._semantic_text_from_item(item=item, title=title, summary=summary, keywords=keywords)
            metadata = self._semantic_metadata_from_item(item, semantic_text)

            episodes.append(
                BuiltEpisode(
                    title=title,
                    summary=summary,
                    episode_type=episode_type,
                    rawlog_ids=rawlog_ids,
                    emotion_signal=item.get("emotion_signal"),
                    importance_score=item.get("importance_score"),
                    keywords=keywords,
                    semantic_text=semantic_text,
                    metadata=metadata,
                )
            )

        return episodes

    def build(self, llm_turns: list[dict], valid_rawlog_ids: set[str]) -> list[BuiltEpisode]:
        if not llm_turns:
            return []

        episodes: list[BuiltEpisode] = []
        for item in self.llm_client.build_episodes(llm_turns):
            rawlog_ids = [rawlog_id for rawlog_id in item.get("rawlog_ids", []) if rawlog_id in valid_rawlog_ids]
            if not rawlog_ids:
                continue

            title = str(item.get("title") or "Untitled episode").strip()
            summary = str(item.get("summary") or "Semantic episode generated from conversation turns.").strip()
            episode_type = str(item.get("episode_type") or "topic").strip()
            keywords = item.get("keywords") if isinstance(item.get("keywords"), list) else None
            semantic_text = self._semantic_text_from_item(
                item=item,
                title=title,
                summary=summary,
                keywords=keywords,
            )
            metadata = self._semantic_metadata_from_item(item, semantic_text)

            episodes.append(
                BuiltEpisode(
                    title=title,
                    summary=summary,
                    episode_type=episode_type,
                    rawlog_ids=rawlog_ids,
                    emotion_signal=item.get("emotion_signal"),
                    importance_score=item.get("importance_score"),
                    keywords=keywords,
                    semantic_text=semantic_text,
                    metadata=metadata,
                )
            )

        return episodes

    def _semantic_text_from_item(
        self,
        item: dict,
        title: str,
        summary: str,
        keywords: list[str] | None,
    ) -> str:
        semantic_text = item.get(SEMANTIC_TEXT_METADATA_KEY)
        if isinstance(semantic_text, str) and semantic_text.strip():
            return semantic_text.strip()

        lines = []
        labels = {
            "user_goal": "User goal",
            "context": "Context",
            "decision_or_insight": "Decision or insight",
            "emotional_or_situational_cue": "Emotional or situational cue",
            "representative_snippets": "Representative snippets",
        }
        for key in SEMANTIC_DETAIL_KEYS:
            value = item.get(key)
            text = self._metadata_value_text(value)
            if text:
                lines.append(f"{labels[key]}: {text}")

        if lines:
            return "\n".join(lines)
        return self._semantic_text(title=title, summary=summary, keywords=keywords)

    def _semantic_metadata_from_item(self, item: dict, semantic_text: str) -> dict:
        metadata = {
            SEMANTIC_TEXT_METADATA_KEY: semantic_text,
            EMBEDDING_SOURCE_VERSION_METADATA_KEY: EMBEDDING_SOURCE_VERSION,
        }
        for key in SEMANTIC_DETAIL_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                metadata[key] = value.strip()
            elif isinstance(value, list):
                cleaned = [str(entry).strip() for entry in value if str(entry).strip()]
                if cleaned:
                    metadata[key] = cleaned
        return metadata

    def _metadata_value_text(self, value: object) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return " | ".join(str(entry).strip() for entry in value if str(entry).strip())
        return ""

    def _semantic_text(self, title: str, summary: str, keywords: list[str] | None) -> str:
        keyword_text = ", ".join(str(keyword) for keyword in keywords or [])
        return "\n".join(
            [
                f"Title: {title.strip()}",
                f"Summary: {summary.strip()}",
                f"Keywords: {keyword_text}",
            ]
        )
