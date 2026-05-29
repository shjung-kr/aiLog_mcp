from app.llm.client import LLMClient
from app.schemas.chat import ChatMessageCreate
from app.services.retrieval_service import RetrievalService
from app.services.rawlog_service import RawLogService
from app.services.session_service import SessionService
from app.services.turn_service import TurnService
from app.services.user_style_service import UserStyleService
from app.utils.datetime import utc_now


class ChatService:
    def __init__(
        self,
        session_service: SessionService,
        rawlog_service: RawLogService,
        llm_client: LLMClient,
        turn_service: TurnService | None = None,
        retrieval_service: RetrievalService | None = None,
        user_style_service: UserStyleService | None = None,
    ) -> None:
        self.session_service = session_service
        self.rawlog_service = rawlog_service
        self.llm_client = llm_client
        self.turn_service = turn_service
        self.retrieval_service = retrieval_service
        self.user_style_service = user_style_service

    def send_message(self, payload: ChatMessageCreate):
        session_id = payload.session_id
        if session_id is None:
            session = self.session_service.create_session(user_id=payload.user_id, title=payload.title)
            session_id = session.session_id
        else:
            self.session_service.require_session(session_id)

        user_message = self.rawlog_service.create_rawlog(
            session_id=session_id,
            sequence_no=self.rawlog_service.get_next_sequence_no(session_id),
            speaker_type="user",
            content=payload.content,
            occurred_at=utc_now(),
            message_type="question",
            metadata=payload.metadata,
        )

        conversation = self.rawlog_service.list_session_rawlogs(session_id)
        memory_context = None
        context_used: list[dict] = []
        if self.retrieval_service is not None:
            # Pass last 2 turns (up to 4 messages) so follow-up queries can be enriched
            recent_turns = [
                f"{r.speaker_type}: {r.content[:300]}"
                for r in conversation[:-1]
                if r.speaker_type in ("user", "assistant")
            ][-4:]
            memory_context, context_used = self.retrieval_service.retrieve_for_query(
                payload.content, session_id=session_id, recent_turns=recent_turns
            )

        user_style = None
        style_updated = False
        if self.user_style_service is not None:
            style_updated = self.user_style_service.consume_update_notification()
            user_style = self.user_style_service.get_style_profile()

        assistant_text, source_model, sources = self.llm_client.generate_reply(
            conversation,
            memory_context=memory_context,
            use_web_search=True,
            user_style=user_style,
        )
        assistant_message = self.rawlog_service.create_rawlog(
            session_id=session_id,
            sequence_no=self.rawlog_service.get_next_sequence_no(session_id),
            speaker_type="assistant",
            content=assistant_text,
            occurred_at=utc_now(),
            message_type="answer",
            source_model=source_model,
            reply_to_rawlog_id=user_message.rawlog_id,
            metadata={
                "sources": sources,
                "context_used": context_used,
                "memory_context_used": bool(context_used),
                "web_search_enabled": True,
            },
        )
        if self.turn_service is not None:
            self.turn_service.create_from_pair(user_message, assistant_message)
        return session_id, user_message, assistant_message, sources, context_used, style_updated
