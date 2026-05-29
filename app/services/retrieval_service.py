import re
from dataclasses import asdict, dataclass
from math import sqrt

# Phrases that indicate recall/search intent rather than the content being searched.
# We strip these so the embedding focuses on the actual topic, not the meta-framing.
_MEMORY_META_RE = re.compile(
    r"(기억\s*하니[?]?|기억\s*해[?]?|기억\s*나니[?]?|기억\s*나[?]?|"
    r"이야기\s*했던\s*(?:것|거)[을를]?|이야기\s*했던[가]?|얘기\s*했던\s*(?:것|거)[을를]?|"
    r"얘기\s*했던[가]?|대화\s*했던|이야기\s*한\s*적|얘기\s*한\s*적|"
    r"했던\s*거\s*기억|했던\s*적\s*있[니]?|너와\s*이야기|우리가\s*이야기|"
    r"(?:예전|이전|지난\s*번)\s*(?:대화\s*)?기록|"
    r"예전(?:에|의)?|이전(?:에|의)?|전에|지난\s*번(?:에)?|그때|"
    r"대화\s*(?:내용|기록|주제)|이야기\s*(?:내용|주제)|얘기\s*(?:내용|주제)|"
    r"주된\s*내용[이은을]?|주요\s*내용[이은을]?|핵심\s*내용[이은을]?|핵심만|핵심을?|"
    r"뭐\s*였(?:지|더라)|뭐였(?:지|더라)|무엇\s*이었(?:지|더라)|"
    r"어떤\s*(?:내용|이야기|얘기)(?:이었(?:지|더라))?|"
    r"다시\s*(?:불러와서|찾아서|검색해서|정리해서)?|"
    r"불러와서|찾아(?:줘|봐)|검색해(?:줘|봐)|알려줘|말해줘|정리해줘|보여줘|"
    r"에\s*대해서|에\s*대해|에\s*관해서|에\s*관해|[?？!~])",
    re.IGNORECASE,
)
_MEMORY_META_REMNANT_RE = re.compile(
    r"(^|\s)(?:거|것|내용[이은을]?|기록|이야기|얘기)(?=\s|$|[.,，。])",
    re.IGNORECASE,
)
_PUNCT_SPACE_RE = re.compile(r"\s+([.,，。])")
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_SUMMARY_INTENT_RE = re.compile(r"(주된\s*내용|주요\s*내용|핵심|요약|정리)", re.IGNORECASE)
_EXACT_INTENT_RE = re.compile(r"(정확히|원문|뭐라고|어떻게\s*말|표현|문장)", re.IGNORECASE)
_LIST_INTENT_RE = re.compile(r"(목록|리스트|나열|정리해\s*줘|정리해줘)", re.IGNORECASE)
_CONTINUATION_INTENT_RE = re.compile(r"(이어서|계속|다음|이어\s*서)", re.IGNORECASE)
_SAYING_META_RE = re.compile(r"(말\s*했(?:지|더라)?|했(?:지|더라))", re.IGNORECASE)

# Episodes whose semantic_text is primarily about a FAILED or EMPTY memory recall attempt
# carry no useful content — they only record that recall didn't work, which is noise for retrieval.
_FAILED_RECALL_RE = re.compile(
    r"(자동으로\s*불러오[진]?\s*못|세부를\s*(바로\s*)?(복원|확인)\s*(하기|하지)\s*(어렵|못)|"
    r"억지로\s*(과거|이전)\s*(내용|기억|맥락)을?\s*(끌어|복원)|"
    r"확답은?\s*피했|확답\s*하기\s*(어렵|힘들)|"
    r"(이전|과거)\s*(대화|내용|기억)을?\s*(직접|바로)\s*(확인|복원)\s*(불가|안\s*됨|할\s*수\s*없)|"
    r"기억\s*(검증|확인)에\s*매달|기억\s*여부를\s*캐묻|"
    # Empty-recall episodes: recorded a recall attempt but fell back to suggesting related topics
    r"관련성이\s*높은\s*주제들을\s*제안|"
    r"이전\s*대화를\s*떠올리며.{0,60}(주제|내용)을?\s*(제안|추천|소개)|"
    r"(맥락이|대화가)\s*(흐릿|불분명|기억나지\s*않))",
    re.DOTALL,
)
META_RECALL_PENALTY = 0.12

from app.db.models.episode import Episode
from app.db.models.search_log import SearchLog
from app.db.repositories.episode_repository import EpisodeRepository
from app.db.repositories.search_repository import SearchRepository
from app.llm.client import LLMClient
from app.services.rawlog_service import RawLogService
from app.utils.datetime import utc_now
from app.utils.ids import new_id

EMBEDDING_METADATA_KEY = "semantic_embedding"
TITLE_EMBEDDING_METADATA_KEY = "title_embedding"
SEMANTIC_TEXT_METADATA_KEY = "semantic_text"
RETRIEVAL_SCORE_THRESHOLD = 0.35
RETRIEVAL_CANDIDATE_LIMIT = 12
KEYWORD_BOOST_WEIGHT = 0.15  # hybrid: 85% embedding + 15% keyword overlap


@dataclass
class RetrievalQueryParse:
    original_query: str
    content_query: str
    recall_intent: bool
    answer_intent: str
    specificity: str
    removed_meta_terms: list[str]
    answer_shape_terms: list[str]


class RetrievalService:
    def __init__(
        self,
        episode_repository: EpisodeRepository,
        rawlog_service: RawLogService,
        llm_client: LLMClient,
        search_repository: SearchRepository | None = None,
    ) -> None:
        self.episode_repository = episode_repository
        self.rawlog_service = rawlog_service
        self.llm_client = llm_client
        self.search_repository = search_repository

    def retrieve_for_query(
        self,
        query: str,
        session_id: str | None = None,
        recent_turns: list[str] | None = None,
    ) -> tuple[str | None, list[dict]]:
        query = query.strip()
        if not query:
            return None, []

        # Step 1: parse recall/meta language, then embed the content-focused query.
        query_parse = self._parse_recall_query(query)
        embedding_query = self._build_embedding_query(query_parse, recent_turns)
        query_embedding = self.llm_client.embed_texts([embedding_query])[0]
        query_tokens = self._tokenize(query_parse.content_query or query)
        candidates: list[tuple[float, Episode]] = []
        for episode in self.episode_repository.list_all(limit=500):
            metadata = episode.metadata_json or {}
            embedding = metadata.get(EMBEDDING_METADATA_KEY)
            if not isinstance(embedding, list):
                continue
            cosine = self._cosine_similarity(query_embedding, [float(v) for v in embedding])
            title_embedding = metadata.get(TITLE_EMBEDDING_METADATA_KEY)
            if isinstance(title_embedding, list):
                cosine_title = self._cosine_similarity(query_embedding, [float(v) for v in title_embedding])
                cosine = max(cosine, cosine_title)
            semantic_text = metadata.get(SEMANTIC_TEXT_METADATA_KEY, "") or ""
            # Skip episodes that only record a failed/empty recall — they contain no useful content.
            if _FAILED_RECALL_RE.search(semantic_text[:300]):
                continue
            keyword_score = self._keyword_overlap(query_tokens, semantic_text)
            score = cosine * (1 - KEYWORD_BOOST_WEIGHT) + keyword_score * KEYWORD_BOOST_WEIGHT
            # Boost episodes that have been promoted to long-term memory.
            if metadata.get("promoted_to_ltm"):
                score += 0.05
            if score >= RETRIEVAL_SCORE_THRESHOLD:
                candidates.append((score, episode))

        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)[:RETRIEVAL_CANDIDATE_LIMIT]
        retrieved_log = [
            {"episode_id": ep.episode_id, "title": ep.title, "score": round(sc, 4)}
            for sc, ep in ranked
        ]

        if not ranked:
            self._log(
                query=query,
                session_id=session_id,
                retrieved=retrieved_log,
                curated=[],
                query_parse=query_parse,
                used_episode_id=None,
                reasoning="no candidates above threshold",
            )
            return None, []

        # Step 2: curator — filter with original query + conversation context
        curator_input = [
            {
                "episode_id": ep.episode_id,
                "semantic_text": self._episode_semantic_text(ep),
                "score": round(sc, 4),
            }
            for sc, ep in ranked
        ]
        relevant_ids, reasoning = self.llm_client.curate_episodes(
            query,
            curator_input,
            conversation_context=recent_turns,
            query_parse=asdict(query_parse),
        )
        relevant_set = set(relevant_ids)

        curated = [(sc, ep) for sc, ep in ranked if ep.episode_id in relevant_set]
        curated_log = [
            {"episode_id": ep.episode_id, "title": ep.title, "score": round(sc, 4)}
            for sc, ep in curated
        ]

        if not curated:
            self._log(
                query=query,
                session_id=session_id,
                retrieved=retrieved_log,
                curated=curated_log,
                query_parse=query_parse,
                used_episode_id=None,
                reasoning=reasoning,
            )
            return None, []

        # Step 3: take only the single best episode (spec: 1턴에 한 episode만 활용)
        best_score, best_episode = curated[0]
        semantic_text = self._episode_semantic_text(best_episode)

        self._log(
            query=query,
            session_id=session_id,
            retrieved=retrieved_log,
            curated=curated_log,
            query_parse=query_parse,
            used_episode_id=best_episode.episode_id,
            reasoning=reasoning,
        )

        context_items = [
            {
                "episode_id": best_episode.episode_id,
                "title": best_episode.title,
                "score": round(best_score, 4),
                "start_at": best_episode.start_at.isoformat(),
                "rawlog_ids": [],
                "recall_intent": query_parse.recall_intent,
            }
        ]
        return semantic_text, context_items

    def _parse_recall_query(self, query: str) -> RetrievalQueryParse:
        removed_meta_terms = [match.group(0).strip() for match in _MEMORY_META_RE.finditer(query)]
        topic = _MEMORY_META_RE.sub(" ", query)
        removed_meta_terms.extend(
            match.group(0).strip() for match in _MEMORY_META_REMNANT_RE.finditer(topic)
        )
        topic = _MEMORY_META_REMNANT_RE.sub(" ", topic)
        for pattern in (
            _SUMMARY_INTENT_RE,
            _EXACT_INTENT_RE,
            _LIST_INTENT_RE,
            _CONTINUATION_INTENT_RE,
            _SAYING_META_RE,
        ):
            topic = pattern.sub(" ", topic)
        topic = _PUNCT_SPACE_RE.sub(r"\1", topic)
        topic = topic.strip(" .,\t\n")
        topic = re.sub(r"\s+", " ", topic).strip()

        content_query = topic if removed_meta_terms and topic != query else query
        answer_shape_terms = self._answer_shape_terms(query)
        token_count = len(_TOKEN_RE.findall(content_query))
        if token_count >= 4:
            specificity = "high"
        elif token_count >= 1:
            specificity = "medium"
        else:
            specificity = "low"

        return RetrievalQueryParse(
            original_query=query,
            content_query=content_query,
            recall_intent=bool(removed_meta_terms),
            answer_intent=self._answer_intent(query),
            specificity=specificity,
            removed_meta_terms=list(dict.fromkeys(term for term in removed_meta_terms if term)),
            answer_shape_terms=answer_shape_terms,
        )

    def _build_embedding_query(self, query_parse: RetrievalQueryParse, recent_turns: list[str] | None) -> str:
        embedding_text = query_parse.content_query or query_parse.original_query

        if not recent_turns:
            return embedding_text
        context = "\n".join(recent_turns[-4:])
        return f"{context}\nuser: {embedding_text}"

    def _answer_intent(self, query: str) -> str:
        if _EXACT_INTENT_RE.search(query):
            return "exact"
        if _LIST_INTENT_RE.search(query):
            return "list"
        if _CONTINUATION_INTENT_RE.search(query):
            return "continuation"
        if _SUMMARY_INTENT_RE.search(query):
            return "summary"
        return "unknown"

    def _answer_shape_terms(self, query: str) -> list[str]:
        terms: list[str] = []
        for pattern in (_SUMMARY_INTENT_RE, _EXACT_INTENT_RE, _LIST_INTENT_RE, _CONTINUATION_INTENT_RE):
            terms.extend(match.group(0).strip() for match in pattern.finditer(query))
        return list(dict.fromkeys(term for term in terms if term))

    def _episode_semantic_text(self, episode: Episode) -> str:
        metadata = episode.metadata_json or {}
        semantic_text = (metadata.get(SEMANTIC_TEXT_METADATA_KEY) or "").strip()

        # Always prepend title + keywords so concrete terms are preserved
        # even if semantic_text paraphrased them into abstract language.
        header_parts = []
        if episode.title:
            header_parts.append(f"주제: {episode.title}")
        if episode.keywords:
            header_parts.append(f"키워드: {', '.join(str(k) for k in episode.keywords)}")
        header = "\n".join(header_parts)

        if semantic_text:
            return f"{header}\n{semantic_text}"
        # fallback when semantic_text is missing
        return f"{header}\n{episode.summary}" if episode.summary else header

    def _log(
        self,
        query: str,
        session_id: str | None,
        retrieved: list[dict],
        curated: list[dict],
        query_parse: RetrievalQueryParse,
        used_episode_id: str | None,
        reasoning: str,
    ) -> None:
        if self.search_repository is None:
            return
        try:
            log = SearchLog(
                log_id=new_id(),
                query=query,
                session_id=session_id,
                retrieved_json=retrieved,
                curated_json=curated,
                query_parse_json=asdict(query_parse),
                used_episode_id=used_episode_id,
                curator_reasoning=reasoning or None,
                created_at=utc_now(),
            )
            self.search_repository.create(log)
        except Exception:
            pass

    def _tokenize(self, text: str) -> set[str]:
        return {t for t in re.findall(r"[0-9A-Za-z가-힣]{2,}", text.lower()) if len(t) >= 2}

    def _keyword_overlap(self, query_tokens: set[str], text: str) -> float:
        if not query_tokens or not text:
            return 0.0
        text_tokens = self._tokenize(text)
        if not text_tokens:
            return 0.0
        matched = query_tokens & text_tokens
        return len(matched) / len(query_tokens)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(lv * rv for lv, rv in zip(left, right))
        left_norm = sqrt(sum(v * v for v in left))
        right_norm = sqrt(sum(v * v for v in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)
