import re
from math import sqrt

from app.db.models.episode import Episode
from app.llm.client import LLMClient
from app.pipeline.episode.episode_builder import (
    EMBEDDING_SOURCE_VERSION,
    EMBEDDING_SOURCE_VERSION_METADATA_KEY,
    SEMANTIC_TEXT_METADATA_KEY,
    EpisodeBuilder,
)
from app.services.episode_service import EpisodeService
from app.services.gist_service import GistService
from app.services.memory_promotion_service import MemoryPromotionService
from app.services.rawlog_service import RawLogService
from app.services.turn_service import TurnService

EMBEDDING_MERGE_THRESHOLD = 0.70
MIN_EMBEDDING_COSINE_FOR_MERGE = 0.62
FALLBACK_MERGE_SIMILARITY_THRESHOLD = 0.16
EMBEDDING_METADATA_KEY = "semantic_embedding"
TITLE_EMBEDDING_METADATA_KEY = "title_embedding"
EMBEDDING_MODEL_METADATA_KEY = "semantic_embedding_model"
STOPWORDS = {
    "turn",
    "사용자",
    "대화",
    "내용",
    "관련",
    "정보",
    "요청",
    "논의",
    "정리",
    "확인",
    "설명",
}

TERM_ALIASES = {
    "episode": "에피소드",
    "episodes": "에피소드",
    "episode_id": "에피소드",
    "episode_type": "에피소드",
    "rawlog": "rawlog",
    "rawlogs": "rawlog",
    "session": "세션",
    "sessions": "세션",
    "source_session_id": "세션",
    "schema": "스키마",
    "metadata": "메타데이터",
}

DOMAIN_TERMS = {"에피소드", "rawlog", "세션", "스키마", "기억구조", "의미", "분류"}


class EpisodeBuilderService:
    def __init__(
        self,
        episode_service: EpisodeService,
        turn_service: TurnService,
        rawlog_service: RawLogService,
        llm_client: LLMClient,
        gist_service: GistService | None = None,
        memory_promotion_service: MemoryPromotionService | None = None,
    ) -> None:
        self.episode_service = episode_service
        self.turn_service = turn_service
        self.rawlog_service = rawlog_service
        self.llm_client = llm_client
        self.gist_service = gist_service
        self.memory_promotion_service = memory_promotion_service
        self.episode_builder = EpisodeBuilder(llm_client)
        self._last_match_score = 0.0
        self._last_match_method: str | None = None

    def build_from_session(self, session_id: str, rebuild_existing: bool = True) -> list[Episode]:
        gists = self.gist_service.list_for_session(session_id) if self.gist_service else []

        # Snapshot the session's own episodes (source_session_id matches) before any modification.
        session_episodes_before: list[Episode] = (
            self.episode_service.list_episodes(source_session_id=session_id, limit=200)
            if rebuild_existing else []
        )
        own_ep_ids_before: set[str] = {ep.episode_id for ep in session_episodes_before}

        # Load full cross-session pool once to avoid duplicate DB round-trips.
        all_cross_episodes: list[Episode] = [
            ep for ep in self.episode_service.list_all_episodes(limit=500)
            if ep.episode_id not in own_ep_ids_before
        ]

        # Cross-session episodes this session previously contributed rawlogs to via merging.
        # Including them as session-preferred candidates prevents duplicate creation on rebuild.
        contributing_episodes: list[Episode] = [
            ep for ep in all_cross_episodes
            if session_id in (ep.metadata_json or {}).get("contributing_session_ids", [])
        ]
        contributing_ep_ids: set[str] = {ep.episode_id for ep in contributing_episodes}

        # Combined set that receives session-preferred merge treatment.
        session_ep_ids_before: set[str] = own_ep_ids_before | contributing_ep_ids

        cross_session_episodes: list[Episode] = [
            ep for ep in all_cross_episodes if ep.episode_id not in contributing_ep_ids
        ]
        existing_episodes: list[Episode] = [
            *session_episodes_before,
            *contributing_episodes,
            *cross_session_episodes,
        ]

        if gists:
            valid_rawlog_ids: set[str] = set()
            gist_segments: list[dict] = []
            for gist in gists:
                rawlog_ids = (gist.metadata_json or {}).get("rawlog_ids", [])
                valid_rawlog_ids.update(rawlog_ids)
                gist_segments.append({
                    "gist_id": gist.gist_id,
                    "gist_text": gist.gist_text,
                    "topic": gist.topic,
                    "intent": gist.intent,
                    "rawlog_ids": rawlog_ids,
                })

            built_episodes = self.episode_builder.build_from_gists(
                gist_segments=gist_segments,
                valid_rawlog_ids=valid_rawlog_ids,
            )
        else:
            turns = self.turn_service.build_from_session(session_id)
            rawlogs = self.rawlog_service.list_session_rawlogs(session_id)
            rawlog_by_id = {rawlog.rawlog_id: rawlog for rawlog in rawlogs}

            llm_turns = []
            for turn in turns:
                start = rawlog_by_id[turn.start_rawlog_id].sequence_no
                end = rawlog_by_id[turn.end_rawlog_id].sequence_no
                turn_rawlogs = [rawlog for rawlog in rawlogs if start <= rawlog.sequence_no <= end]
                llm_turns.append({
                    "turn_id": turn.turn_id,
                    "rawlogs": [
                        {
                            "rawlog_id": rawlog.rawlog_id,
                            "sequence_no": rawlog.sequence_no,
                            "speaker_type": rawlog.speaker_type,
                            "content": rawlog.content,
                        }
                        for rawlog in turn_rawlogs
                    ],
                })

            if not llm_turns:
                return []

            valid_rawlog_ids = {rawlog.rawlog_id for rawlog in rawlogs}
            built_episodes = self.episode_builder.build(llm_turns=llm_turns, valid_rawlog_ids=valid_rawlog_ids)

        # Freeze the merge candidate pool so episodes created during this run
        # cannot become unintended merge targets for later items in the same loop.
        merge_candidate_pool: list[Episode] = list(existing_episodes)

        episodes: list[Episode] = []
        matched_session_ep_ids: set[str] = set()

        for item in built_episodes:
            matching_episode = self._find_matching_episode(
                existing_episodes=merge_candidate_pool,
                title=item.title,
                summary=item.summary,
                episode_type=item.episode_type,
                keywords=item.keywords,
                semantic_text=item.semantic_text,
                session_ep_ids=session_ep_ids_before,
            )
            if matching_episode is None:
                # Rawlog-range fallback prevents UNIQUE constraint violations when
                # semantic similarity was too low to detect an existing episode.
                matching_episode = self._find_episode_by_rawlog_range(item.rawlog_ids)

            if matching_episode is None:
                episode = self.episode_service.create_episode(
                    title=item.title,
                    summary=item.summary,
                    episode_type=item.episode_type,
                    rawlog_ids=item.rawlog_ids,
                    emotion_signal=item.emotion_signal,
                    importance_score=item.importance_score,
                    source_session_id=session_id,
                    keywords=item.keywords,
                    metadata={
                        "builder": "llm_episode_builder_v1",
                        "episode_scope": "cross_session_semantic",
                        **item.metadata,
                    },
                )
                self._store_episode_embedding(episode)
                # Add to existing_episodes for rawlog-range uniqueness tracking only.
                # Do NOT add to merge_candidate_pool — the pool is frozen for this run.
                existing_episodes.append(episode)
            else:
                if matching_episode.episode_id in session_ep_ids_before:
                    matched_session_ep_ids.add(matching_episode.episode_id)
                old_semantic_text = self._episode_semantic_text(matching_episode)
                episode = self.episode_service.merge_episode(
                    episode=matching_episode,
                    title=item.title,
                    summary=item.summary,
                    episode_type=item.episode_type,
                    rawlog_ids=item.rawlog_ids,
                    emotion_signal=item.emotion_signal,
                    importance_score=item.importance_score,
                    keywords=item.keywords,
                    metadata={
                        "builder": "llm_episode_builder_v1",
                        "episode_scope": "cross_session_semantic",
                        "merge_similarity": self._last_match_score,
                        "merge_similarity_method": self._last_match_method,
                        **item.metadata,
                    },
                )
                merged_semantic_text = self._merged_semantic_text(old_semantic_text, item.semantic_text)
                episode.metadata_json = {
                    **(episode.metadata_json or {}),
                    SEMANTIC_TEXT_METADATA_KEY: merged_semantic_text,
                }
                self.episode_service.update_episode(episode)
                self._store_episode_embedding(episode)
                # Refresh the merged episode in the candidate pool so subsequent items
                # compare against the updated semantic embedding, not the pre-merge version.
                for idx, pool_ep in enumerate(merge_candidate_pool):
                    if pool_ep.episode_id == episode.episode_id:
                        merge_candidate_pool[idx] = episode
                        break
            episodes.append(episode)

        # Remove only the session's OWN stale episodes — contributing cross-session
        # episodes are shared with other sessions and must not be deleted here.
        stale_ids = own_ep_ids_before - matched_session_ep_ids
        for ep_id in stale_ids:
            try:
                self.episode_service.delete_episode(ep_id)
            except Exception:
                pass

        # Promote eligible episodes to long-term memory.
        if self.memory_promotion_service and episodes:
            try:
                self.memory_promotion_service.promote_from_episodes(episodes)
            except Exception:
                pass

        return episodes

    def _find_episode_by_rawlog_range(self, rawlog_ids: list[str]) -> Episode | None:
        if not rawlog_ids:
            return None
        try:
            rawlogs = self.rawlog_service.list_rawlogs_by_ids(rawlog_ids)
            if not rawlogs:
                return None
            ordered = sorted(rawlogs, key=lambda r: (r.occurred_at, r.sequence_no))
            return self.episode_service.get_by_rawlog_range(
                ordered[0].rawlog_id, ordered[-1].rawlog_id
            )
        except Exception:
            return None

    def _merged_semantic_text(self, old_text: str, new_text: str) -> str:
        if not old_text.strip():
            return new_text
        if not new_text.strip():
            return old_text
        try:
            return self.llm_client.merge_semantic_text(old_text, new_text)
        except Exception:
            return new_text

    def _find_matching_episode(
        self,
        existing_episodes: list[Episode],
        title: str,
        summary: str,
        episode_type: str,
        keywords: list[str] | None,
        semantic_text: str,
        session_ep_ids: set[str] | None = None,
    ) -> Episode | None:
        self._last_match_score = 0.0
        self._last_match_method = None
        try:
            candidate_embedding = self.llm_client.embed_texts([semantic_text])[0]
            best_session_episode = None
            best_session_score = 0.0
            best_cross_episode = None
            best_cross_score = 0.0

            for episode in existing_episodes:
                episode_embedding = self._ensure_episode_embedding(episode)
                cosine_score = self._cosine_similarity(candidate_embedding, episode_embedding)

                # Title embedding backup: recovers matches where semantic_text formats differ
                # (e.g. legacy Title/Summary/Keywords fallback vs. LLM narrative style).
                title_embedding = (episode.metadata_json or {}).get(TITLE_EMBEDDING_METADATA_KEY)
                if isinstance(title_embedding, list):
                    title_cosine = self._cosine_similarity(
                        candidate_embedding, [float(v) for v in title_embedding]
                    )
                    cosine_score = max(cosine_score, title_cosine * 0.92)

                if cosine_score < MIN_EMBEDDING_COSINE_FOR_MERGE:
                    continue
                score = self._merge_score(
                    embedding_cosine=cosine_score,
                    left_keywords=episode.keywords,
                    right_keywords=keywords,
                    left_episode_type=episode.episode_type,
                    right_episode_type=episode_type,
                )
                if session_ep_ids and episode.episode_id in session_ep_ids:
                    if score > best_session_score:
                        best_session_score = score
                        best_session_episode = episode
                else:
                    if score > best_cross_score:
                        best_cross_score = score
                        best_cross_episode = episode

            # Prefer same-session episode when it meets the merge threshold.
            # This prevents existing session episodes from being incorrectly marked
            # stale just because a cross-session episode scored marginally higher.
            if best_session_score >= EMBEDDING_MERGE_THRESHOLD:
                self._last_match_score = best_session_score
                self._last_match_method = "semantic_embedding_hybrid_session_preferred"
                return best_session_episode
            if best_cross_score >= EMBEDDING_MERGE_THRESHOLD:
                self._last_match_score = best_cross_score
                self._last_match_method = "semantic_embedding_hybrid"
                return best_cross_episode

            self._last_match_score = max(best_session_score, best_cross_score)
            self._last_match_method = "semantic_embedding_hybrid"
            return None
        except Exception:
            return self._find_matching_episode_by_metadata(
                existing_episodes=existing_episodes,
                title=title,
                summary=summary,
                keywords=keywords,
            )

    def _find_matching_episode_by_metadata(
        self,
        existing_episodes: list[Episode],
        title: str,
        summary: str,
        keywords: list[str] | None,
    ) -> Episode | None:
        best_episode = None
        best_score = 0.0
        for episode in existing_episodes:
            score = self._episode_similarity(
                episode.title,
                episode.summary,
                episode.keywords,
                title,
                summary,
                keywords,
            )
            if score > best_score:
                best_score = score
                best_episode = episode

        self._last_match_score = best_score
        self._last_match_method = "metadata_fallback"
        if best_score < FALLBACK_MERGE_SIMILARITY_THRESHOLD:
            return None
        return best_episode

    def _store_episode_embedding(self, episode: Episode) -> None:
        semantic_text = self._episode_semantic_text(episode)
        title_text = self._title_index_text(episode)
        embeddings = self.llm_client.embed_texts([semantic_text, title_text])
        episode.metadata_json = {
            **(episode.metadata_json or {}),
            SEMANTIC_TEXT_METADATA_KEY: semantic_text,
            EMBEDDING_METADATA_KEY: embeddings[0],
            TITLE_EMBEDDING_METADATA_KEY: embeddings[1],
            EMBEDDING_MODEL_METADATA_KEY: getattr(self.llm_client, "embedding_model", None),
            EMBEDDING_SOURCE_VERSION_METADATA_KEY: EMBEDDING_SOURCE_VERSION,
        }
        self.episode_service.update_episode(episode)

    def _ensure_episode_embedding(self, episode: Episode) -> list[float]:
        metadata = episode.metadata_json or {}
        embedding = metadata.get(EMBEDDING_METADATA_KEY)
        semantic_text = self._episode_semantic_text(episode)
        if (
            isinstance(embedding, list)
            and metadata.get(SEMANTIC_TEXT_METADATA_KEY) == semantic_text
            and metadata.get(EMBEDDING_SOURCE_VERSION_METADATA_KEY) == EMBEDDING_SOURCE_VERSION
        ):
            return [float(value) for value in embedding]

        embedding = self.llm_client.embed_texts([semantic_text])[0]
        episode.metadata_json = {
            **metadata,
            SEMANTIC_TEXT_METADATA_KEY: semantic_text,
            EMBEDDING_METADATA_KEY: embedding,
            EMBEDDING_MODEL_METADATA_KEY: getattr(self.llm_client, "embedding_model", None),
            EMBEDDING_SOURCE_VERSION_METADATA_KEY: EMBEDDING_SOURCE_VERSION,
        }
        self.episode_service.update_episode(episode)
        return embedding

    def _episode_semantic_text(self, episode: Episode) -> str:
        metadata = episode.metadata_json or {}
        semantic_text = (metadata.get(SEMANTIC_TEXT_METADATA_KEY) or "").strip()
        if semantic_text:
            return semantic_text
        return self._semantic_text(title=episode.title, summary=episode.summary, keywords=episode.keywords)

    def _title_index_text(self, episode: Episode) -> str:
        """Short title+keywords text for topic recall queries (complements long semantic_text embedding)."""
        parts = [episode.title.strip()]
        if episode.keywords:
            parts.append(", ".join(str(k) for k in episode.keywords))
        return ". ".join(parts)

    def _semantic_text(self, title: str, summary: str, keywords: list[str] | None) -> str:
        keyword_text = ", ".join(str(keyword) for keyword in keywords or [])
        return "\n".join(
            [
                f"Title: {title.strip()}",
                f"Summary: {summary.strip()}",
                f"Keywords: {keyword_text}",
            ]
        )

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _merge_score(
        self,
        embedding_cosine: float,
        left_keywords: list[str] | None,
        right_keywords: list[str] | None,
        left_episode_type: str,
        right_episode_type: str,
    ) -> float:
        keyword_score = self._jaccard(self._keyword_set(left_keywords), self._keyword_set(right_keywords))
        type_score = 1.0 if left_episode_type.strip() == right_episode_type.strip() else 0.0
        return (embedding_cosine * 0.8) + (keyword_score * 0.1) + (type_score * 0.1)

    def _episode_similarity(
        self,
        left_title: str,
        left_summary: str,
        left_keywords: list[str] | None,
        right_title: str,
        right_summary: str,
        right_keywords: list[str] | None,
    ) -> float:
        left_keywords_set = self._keyword_set(left_keywords)
        right_keywords_set = self._keyword_set(right_keywords)
        keyword_score = self._jaccard(left_keywords_set, right_keywords_set)

        left_terms = self._term_set(f"{left_title} {left_summary}")
        right_terms = self._term_set(f"{right_title} {right_summary}")
        term_score = self._jaccard(left_terms, right_terms)

        left_grams = self._char_grams(f"{left_title} {left_summary}")
        right_grams = self._char_grams(f"{right_title} {right_summary}")
        gram_score = self._jaccard(left_grams, right_grams)
        domain_score = self._domain_score(
            left_keywords_set | left_terms,
            right_keywords_set | right_terms,
        )

        return (keyword_score * 0.4) + (term_score * 0.25) + (gram_score * 0.15) + (domain_score * 0.2)

    def _keyword_set(self, keywords: list[str] | None) -> set[str]:
        return {self._normalize_term(str(keyword)) for keyword in keywords or [] if str(keyword).strip()}

    def _term_set(self, text: str) -> set[str]:
        terms = set(re.findall(r"[0-9A-Za-z가-힣_]{2,}", text.lower()))
        return {self._normalize_term(term) for term in terms if self._normalize_term(term) not in STOPWORDS}

    def _char_grams(self, text: str) -> set[str]:
        normalized = re.sub(r"\s+", "", text.lower())
        return {normalized[index : index + 3] for index in range(max(len(normalized) - 2, 0))}

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _domain_score(self, left: set[str], right: set[str]) -> float:
        shared_domain_terms = (left & right) & DOMAIN_TERMS
        if not shared_domain_terms:
            return 0.0
        return min(1.0, len(shared_domain_terms) / 2)

    def _normalize_term(self, term: str) -> str:
        normalized = term.lower().strip()
        return TERM_ALIASES.get(normalized, normalized)
