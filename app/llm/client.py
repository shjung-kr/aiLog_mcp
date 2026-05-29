import json

from openai import OpenAI

from app.core.config import settings
from app.db.models.rawlog import RawLog


SYSTEM_PROMPT = (
    "You are aiLog, a personal AI assistant. "
    "You are NOT ChatGPT. Never describe yourself as ChatGPT, never reference OpenAI help pages "
    "or ChatGPT's memory settings. You are a distinct assistant. "
    "\n\n"
    "You have a long-term memory system that stores past conversations. "
    "Treat the two cases below as HARD rules — no exceptions:\n\n"
    "CASE A — User references a past conversation (e.g. '이야기했던 것 같은데', '말했던 거', "
    "'기억해?', 'we discussed', 'you mentioned'):\n"
    "  • If background memory context IS provided: explicitly confirm you have a record, "
    "    then immediately lead with a specific detail from the retrieved context. "
    "    The confirmation should make clear this comes from an actual past conversation, not general knowledge "
    "    (e.g. '응, 얘기했었어. [specific retrieved detail].' / '맞아, 그때 [specific point from memory].'). "
    "    Do NOT give a generic topic explanation — every sentence should be grounded in the retrieved details. "
    "    If the retrieved context is sparse, say so honestly rather than padding with general knowledge.\n"
    "  • If background memory context is NOT provided: state clearly and briefly that you have "
    "    no record of that conversation — then STOP. Do NOT explain the topic in general, "
    "    do NOT produce plausible-sounding content about the subject, do NOT use web search "
    "    to fill the gap. One honest sentence is enough (e.g. '이온채널 관련 이전 대화 기록을 "
    "    찾지 못했어요. 다시 이야기해주시면 같이 살펴볼게요.'). "
    "    General topic knowledge is irrelevant here — the user wants to know if YOU remember, "
    "    not a textbook explanation.\n\n"
    "CASE B — Normal conversation (user is NOT asking about a past conversation): "
    "If background memory context is provided, weave relevant details naturally into your reply "
    "without flagging it as memory retrieval. Phrases like '기억하고 있어요', '지난번에 말씀하신', "
    "'I remember' are forbidden here. If the context is not relevant, ignore it entirely.\n\n"
    "CRITICAL: Only use information from background memory context to describe past conversations. "
    "Never reconstruct, guess, or fabricate what was discussed.\n\n"
    "Be direct and conversational. Lead with the answer, not a preamble. "
    "Do not ask for permission to answer — just answer. "
    "Avoid phrases like '원하시면 ~해드릴게요' when the user has already asked something specific. "
    "Do not mention RawLog, Turn, Episode, retrieval, or internal architecture. "
    "If web search is used, ground factual claims in the searched sources. "
    "Match the user's language and tone naturally."
)

_SEMANTIC_TEXT_INSTRUCTION = (
    "semantic_text is the most important field: it is both the embedding index and the context injection material. "
    "It must be optimized for recall search while still being usable as injected memory context. "
    "The first sentence must be a search-focused topic sentence: name the concrete topic, entities, concepts, "
    "objects, product names, domain terms, and likely recall phrases a user might later ask about. "
    "CRITICAL: preserve the exact Korean terms and nouns from the conversation (e.g. '특허', '특허성', '특허 출원' "
    "must appear verbatim if the conversation used them — do not paraphrase to '차별점' or '포지셔닝'). "
    "Do not let the first sentence be dominated by meta-recall framing such as remembering, checking memory, "
    "asking whether a past conversation happened, or summarizing that a conversation occurred. "
    "After the first sentence, write a first-person retrospective narrative — the kind of internal monologue "
    "a person would write in a personal log: '~을 시도했는데 ~해서 ~로 바꿨다', '~가 문제였고 그래서 ~를 결정했다'. "
    "Include what the user was trying to do, what happened or was discovered, why a decision was made, "
    "and what changed as a result. If the conversation is about a domain topic, keep the domain topic "
    "as the semantic center instead of the act of recalling or searching. "
    "Forbidden styles in semantic_text: third-person observation ('사용자가 ~를 했다'), "
    "fact-card lists (bullet-point style), meta-commentary ('이 에피소드는 ~에 관한 것이다'), "
    "and empty memory-status entries whose main content is only that memory was unclear or unavailable. "
    "Target length: 60-150 tokens. Dense and self-contained. "
)


class LLMClient:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.embedding_model = settings.openai_embedding_model

    def analyze_user_style(
        self,
        user_messages: list[str],
        existing_profile: dict | None = None,
    ) -> dict | None:
        existing_section = ""
        if existing_profile:
            existing_section = (
                f"\n\nCurrently known profile (refine, don't reset):\n"
                f"{json.dumps(existing_profile, ensure_ascii=False)}"
            )

        response = self.client.responses.create(
            model=self.model,
            store=False,
            instructions=(
                "You are a communication style analyst. "
                "Analyze the user's messages and extract their communication style as structured JSON. "
                "Return ONLY valid JSON with exactly these keys:\n"
                '{\n'
                '  "tone": "어투 설명 (e.g. 반말·간결체, 존댓말·격식체)",\n'
                '  "logic_structure": "논리 전개 방식 (e.g. 결론 먼저 → 근거 나열)",\n'
                '  "vocabulary": ["자주 쓰는 도메인 용어나 표현들"],\n'
                '  "response_preference": "선호하는 응답 형식 (e.g. 간결한 서술, 개조식 목록)",\n'
                '  "domain_expertise": ["전문성이 보이는 분야들"]\n'
                "}\n"
                "Base your analysis on patterns across ALL provided messages, not individual messages. "
                "If a field is unclear from the messages, make a conservative best guess. "
                "Write field values in Korean."
                f"{existing_section}"
            ),
            input=(
                f"Analyze the communication style from these {len(user_messages)} user messages:\n\n"
                + "\n---\n".join(user_messages[-30:])
            ),
        )
        content = (response.output_text or "").strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end < start:
                return None
            try:
                data = json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                return None
        return data if isinstance(data, dict) else None

    def generate_reply(
        self,
        rawlogs: list[RawLog],
        memory_context: str | None = None,
        use_web_search: bool = True,
        user_style: dict | None = None,
    ) -> tuple[str, str, list[dict]]:
        instructions = SYSTEM_PROMPT
        if user_style:
            style_lines = []
            if user_style.get("tone"):
                style_lines.append(f"어투: {user_style['tone']}")
            if user_style.get("logic_structure"):
                style_lines.append(f"논리 전개: {user_style['logic_structure']}")
            if user_style.get("response_preference"):
                style_lines.append(f"응답 형식: {user_style['response_preference']}")
            if user_style.get("domain_expertise"):
                style_lines.append(f"전문 도메인: {', '.join(user_style['domain_expertise'])}")
            if style_lines:
                instructions = (
                    f"{instructions}\n\n"
                    "--- User style profile (반드시 이 스타일에 맞춰 응답할 것) ---\n"
                    + "\n".join(style_lines)
                )
        if memory_context:
            instructions = (
                f"{instructions}\n\n"
                "--- Background memory context ---\n"
                "This is a verified record of a past conversation. "
                "For CASE A (user referencing a past conversation): use this context as the basis of your answer — "
                "do NOT say you have no record when this context is present. "
                "For CASE B (normal conversation): weave relevant details naturally without flagging it as memory.\n"
                f"{memory_context}"
            )

        request = {
            "model": self.model,
            "instructions": instructions,
            "store": False,  # Do not store on OpenAI side; aiLog manages its own memory
            "input": [
                {
                    "role": rawlog.speaker_type,
                    "content": rawlog.content,
                }
                for rawlog in rawlogs
                if rawlog.speaker_type in {"user", "assistant", "system"}
            ],
        }
        if use_web_search:
            request["tools"] = [{"type": "web_search"}]
            request["tool_choice"] = "auto"
            request["include"] = ["web_search_call.action.sources"]

        response = self.client.responses.create(**request)
        content = (response.output_text or "").strip()
        if not content:
            raise RuntimeError("OpenAI returned an empty response")
        return content, self.model, self._extract_sources(response)

    def curate_episodes(
        self,
        query: str,
        candidates: list[dict],
        conversation_context: list[str] | None = None,
        query_parse: dict | None = None,
    ) -> tuple[list[str], str]:
        """
        Curator step: filter N embedding candidates down to genuinely contextually relevant ones.
        Returns (relevant_episode_ids, reasoning).
        Conservative by design — when uncertain, exclude.
        conversation_context: recent turns (speaker: content) to help judge follow-up messages.
        """
        if not candidates:
            return [], ""

        candidates_text = "\n\n".join(
            f"[episode_id: {c['episode_id']}]\n{c['semantic_text']}"
            for c in candidates
        )

        context_section = ""
        if conversation_context:
            context_section = (
                "\n\nRecent conversation context (for understanding follow-up messages):\n"
                + "\n".join(conversation_context[-4:])
            )

        parse_section = ""
        if query_parse:
            parse_section = (
                "\n\nStructured retrieval query parse:\n"
                f"{json.dumps(query_parse, ensure_ascii=False)}"
            )

        response = self.client.responses.create(
            model=self.model,
            store=False,
            instructions=(
                "You are a memory relevance curator for a personal AI assistant. "
                "Given the user's current utterance and candidate memory episodes, "
                "decide which episodes are genuinely relevant to what the user is talking about RIGHT NOW. "
                "Use the structured retrieval query parse when present: content_query is the searchable topic, "
                "while answer_intent describes the requested response shape and should not by itself make an episode relevant. "
                "If the current utterance is a follow-up (e.g. '더 설명해줘', '계속해줘', '아까 말한 거'), "
                "use the recent conversation context to infer the actual topic being discussed. "
                "IMPORTANT — memory-recall questions (e.g. '기억나니?', '이야기한 적 있지', '대화한 것 기억해?'): "
                "select episodes that contain INFORMATION ABOUT THE TOPIC being asked about. "
                "These are exactly the cases where memory retrieval matters most — do NOT exclude on uncertainty. "
                "For all other queries: be conservative and exclude when uncertain. "
                "A false positive (irrelevant episode injected) damages trust more than a false negative. "
                "Return JSON only: {\"relevant_ids\": [\"id1\", ...], \"reasoning\": \"one sentence\"}"
            ),
            input=(
                f"User's current utterance:\n{query}"
                f"{parse_section}"
                f"{context_section}\n\n"
                f"Candidate memory episodes:\n{candidates_text}\n\n"
                "Return the episode_ids that are genuinely relevant. "
                "If this is a memory-recall question, include any episode that contains information about the recalled topic. "
                "JSON only."
            ),
        )
        content = (response.output_text or "").strip()
        return self._parse_curator_response(content)

    def _parse_curator_response(self, content: str) -> tuple[list[str], str]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end < start:
                return [], ""
            try:
                data = json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return [], ""
        relevant_ids = data.get("relevant_ids", [])
        reasoning = str(data.get("reasoning", "") or "")
        if not isinstance(relevant_ids, list):
            return [], reasoning
        return [str(id_) for id_ in relevant_ids if id_], reasoning

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def _extract_sources(self, response) -> list[dict]:
        sources: list[dict] = []
        seen_urls: set[str] = set()

        for output in getattr(response, "output", []) or []:
            action = getattr(output, "action", None)
            for source in getattr(action, "sources", []) or []:
                url = getattr(source, "url", None)
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({"url": url, "title": getattr(source, "title", None)})

            for content in getattr(output, "content", []) or []:
                for annotation in getattr(content, "annotations", []) or []:
                    url = getattr(annotation, "url", None)
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        sources.append({"url": url, "title": getattr(annotation, "title", None)})

        return sources

    def merge_semantic_text(self, text_a: str, text_b: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            store=False,
            instructions=(
                "You are aiLog's episode semantic text merger. "
                "Given two semantic texts from related episodes being merged, "
                "synthesize a single semantic_text that preserves the search-focused first sentence "
                "and captures the combined user goals, decisions, context, and key insights from both. "
                "The first sentence must name the concrete searchable topics, entities, concepts, "
                "and likely recall phrases. Then write as a personal log entry: '~을 시도했는데 ~해서 ~로 바꿨다' style. "
                "CRITICAL: preserve exact Korean nouns and domain terms from both inputs verbatim "
                "(e.g. '특허', '특허성', '특허 출원' must not be paraphrased to '차별점' or '포지셔닝'). "
                "Be dense and precise. Synthesize — do not list or concatenate. "
                "Write in the same language as the input. If the input is Korean, respond in Korean. "
                "Target length: 60-150 tokens. "
                "Return only the merged semantic text, no JSON, no labels, no preamble."
            ),
            input=f"Merge these two episode semantic texts into one:\n\nA:\n{text_a}\n\nB:\n{text_b}",
        )
        result = (response.output_text or "").strip()
        if not result:
            raise RuntimeError("OpenAI returned empty merged semantic text")
        return result

    def build_episodes(self, turns: list[dict]) -> list[dict]:
        response = self.client.responses.create(
            model=self.model,
            store=False,
            instructions=(
                "You are aiLog's semantic episode builder. "
                "Read conversation turns and group turns that share the same user goal, topic, problem, or context. "
                "Do not group by connective words alone. Do not copy full conversation text into summaries. "
                "Separate display metadata from embedding evidence. "
                "title, summary, keywords, and episode_type are for UI and coarse filtering. "
                "IMPORTANT — skip these turn types entirely, do NOT create episodes for them: "
                "(1) turns where the assistant says it cannot recall or failed to retrieve past conversations; "
                "(2) turns that are only about checking whether a past conversation happened; "
                "(3) meta-commentary about the memory system itself unless it contains a concrete design decision. "
                "Only create episodes for turns that contain actual content: facts learned, decisions made, "
                "problems solved, topics explained, or insights reached. "
                f"{_SEMANTIC_TEXT_INSTRUCTION}"
                "Write all text fields in the same language as the conversation. If the conversation is in Korean, "
                "write title, summary, keywords, semantic_text, and all other text fields in Korean. "
                "Return JSON only with this shape: "
                "{\"episodes\":[{\"title\":\"...\",\"summary\":\"...\",\"episode_type\":\"topic\","
                "\"emotion_signal\":null,\"importance_score\":0.0,\"keywords\":[\"...\"],"
                "\"user_goal\":\"...\",\"context\":\"...\",\"decision_or_insight\":\"...\","
                "\"emotional_or_situational_cue\":null,\"representative_snippets\":[\"...\"],"
                "\"semantic_text\":\"...\","
                "\"rawlog_ids\":[\"...\"]}]}. "
                "Each rawlog_id must come from the provided turns. Preserve source rawlog_ids exactly."
            ),
            input=(
                "Build semantic episodes from these turns. "
                "A summary should describe the shared context in one concise sentence, not reproduce the dialog. "
                "semantic_text must start with a search-focused topic sentence, then continue as a first-person "
                "retrospective narrative optimized for later natural-language recall queries such as vague references, "
                "remembered situations, prior insights, or domain topic lookups.\n\n"
                f"{json.dumps({'turns': turns}, ensure_ascii=False)}"
            ),
        )
        content = (response.output_text or "").strip()
        if not content:
            raise RuntimeError("OpenAI returned an empty episode build response")
        return self._parse_episode_json(content)

    def build_gist(self, segment_turns: list[dict]) -> dict:
        response = self.client.responses.create(
            model=self.model,
            store=False,
            instructions=(
                "You are aiLog's semantic gist extractor. "
                "Given conversation turns, produce a compact semantic summary capturing what the user was trying "
                "to accomplish, what was decided or learned, and key context. "
                "Do not reproduce dialogue. Be dense and precise. "
                "Write all text fields in the same language as the conversation. "
                "If the conversation is in Korean, write title, gist_text, topic, and intent in Korean. "
                "Return JSON only: "
                "{\"title\": \"...\", \"gist_text\": \"...\", \"topic\": \"...\", "
                "\"intent\": \"...\", \"confidence\": 0.0}"
            ),
            input=(
                "Extract a gist from these conversation turns:\n\n"
                f"{json.dumps({'turns': segment_turns}, ensure_ascii=False)}"
            ),
        )
        content = (response.output_text or "").strip()
        if not content:
            raise RuntimeError("OpenAI returned an empty gist response")
        return self._parse_gist_json(content)

    def _parse_gist_json(self, content: str) -> dict:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end < start:
                raise RuntimeError("OpenAI did not return valid JSON for gist") from None
            data = json.loads(content[start : end + 1])
        if not isinstance(data, dict):
            raise RuntimeError("Gist JSON must be an object")
        return data

    def build_episodes_from_gists(self, gist_segments: list[dict]) -> list[dict]:
        response = self.client.responses.create(
            model=self.model,
            store=False,
            instructions=(
                "You are aiLog's semantic episode builder. "
                "Read gist segments — each is a compressed summary of a conversation chunk — "
                "and group gists that share the same user goal, topic, problem, or context into episodes. "
                "Do not group by connective words alone. "
                "title, summary, keywords, and episode_type are for UI and coarse filtering. "
                "IMPORTANT — skip these gist types entirely, do NOT create episodes for them: "
                "(1) gists where the assistant says it cannot recall or failed to retrieve past conversations; "
                "(2) gists that are only about checking whether a past conversation happened; "
                "(3) meta-commentary about the memory system itself unless it contains a concrete design decision. "
                "Only create episodes for gists that contain actual content: facts learned, decisions made, "
                "problems solved, topics explained, or insights reached. "
                f"{_SEMANTIC_TEXT_INSTRUCTION}"
                "Write all text fields in the same language as the gist content. "
                "If the gists are in Korean, write title, summary, keywords, semantic_text, and all other text fields in Korean. "
                "Return JSON only with this shape: "
                "{\"episodes\":[{\"title\":\"...\",\"summary\":\"...\",\"episode_type\":\"topic\","
                "\"emotion_signal\":null,\"importance_score\":0.0,\"keywords\":[\"...\"],"
                "\"user_goal\":\"...\",\"context\":\"...\",\"decision_or_insight\":\"...\","
                "\"emotional_or_situational_cue\":null,\"representative_snippets\":[\"...\"],"
                "\"semantic_text\":\"...\","
                "\"rawlog_ids\":[\"...\"]}]}. "
                "Each rawlog_id must come from the provided gist segments' rawlog_ids exactly."
            ),
            input=(
                "Build semantic episodes from these gist segments. "
                "Each gist is a compressed summary of a conversation chunk. "
                "Group gists that share the same theme, goal, or problem into a single episode. "
                "semantic_text must start with a search-focused topic sentence, then continue as a first-person "
                "retrospective narrative optimized for natural-language recall queries.\n\n"
                f"{json.dumps({'gists': gist_segments}, ensure_ascii=False)}"
            ),
        )
        content = (response.output_text or "").strip()
        if not content:
            raise RuntimeError("OpenAI returned an empty episode build response from gists")
        return self._parse_episode_json(content)

    def _parse_episode_json(self, content: str) -> list[dict]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end < start:
                raise RuntimeError("OpenAI did not return valid JSON") from None
            data = json.loads(content[start : end + 1])

        episodes = data.get("episodes")
        if not isinstance(episodes, list):
            raise RuntimeError("Episode build JSON must contain an episodes list")
        return [episode for episode in episodes if isinstance(episode, dict)]
