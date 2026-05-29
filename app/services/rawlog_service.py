from app.db.models.rawlog import RawLog
from app.db.repositories.rawlog_repository import RawLogRepository
from app.services.session_service import SessionService
from app.utils.datetime import utc_now
from app.utils.ids import new_id


class RawLogService:
    def __init__(self, rawlog_repository: RawLogRepository, session_service: SessionService) -> None:
        self.rawlog_repository = rawlog_repository
        self.session_service = session_service

    def create_rawlog(
        self,
        session_id: str,
        sequence_no: int,
        speaker_type: str,
        content: str,
        occurred_at,
        message_type: str | None = None,
        reply_to_rawlog_id: str | None = None,
        source_model: str | None = None,
        metadata: dict | None = None,
    ) -> RawLog:
        self.session_service.require_session(session_id)
        self._validate_rawlog_fields(sequence_no=sequence_no, speaker_type=speaker_type, content=content)
        self._validate_sequence(session_id=session_id, sequence_no=sequence_no)

        rawlog = RawLog(
            rawlog_id=new_id(),
            session_id=session_id,
            sequence_no=sequence_no,
            speaker_type=speaker_type,
            content=content,
            occurred_at=occurred_at,
            message_type=message_type,
            reply_to_rawlog_id=reply_to_rawlog_id,
            source_model=source_model,
            stored_at=utc_now(),
            metadata_json=metadata,
        )
        created = self.rawlog_repository.create(rawlog)
        self.session_service.update_last_activity(session_id=session_id, occurred_at=occurred_at)
        return created

    def list_session_rawlogs(self, session_id: str) -> list[RawLog]:
        self.session_service.require_session(session_id)
        return self.rawlog_repository.list_by_session_id(session_id)

    def list_rawlogs_by_ids(self, rawlog_ids: list[str]) -> list[RawLog]:
        return self.rawlog_repository.list_by_ids(rawlog_ids)

    def get_next_sequence_no(self, session_id: str) -> int:
        self.session_service.require_session(session_id)
        return self.rawlog_repository.get_next_sequence_no(session_id)

    def _validate_rawlog_fields(self, sequence_no: int, speaker_type: str, content: str) -> None:
        if sequence_no < 1:
            raise ValueError("sequence_no must be greater than or equal to 1")
        if speaker_type not in {"user", "assistant", "system"}:
            raise ValueError("speaker_type must be one of: user, assistant, system")
        if not content:
            raise ValueError("content must not be empty")

    def _validate_sequence(self, session_id: str, sequence_no: int) -> None:
        latest = self.rawlog_repository.get_latest_for_session(session_id)
        expected = 1 if latest is None else latest.sequence_no + 1
        if sequence_no != expected:
            raise ValueError(f"sequence_no must be {expected} for session '{session_id}'")
