from app.db.models.rawlog import RawLog
from app.db.models.turn import Turn
from app.db.repositories.turn_repository import TurnRepository
from app.services.rawlog_service import RawLogService
from app.utils.ids import new_id


class TurnService:
    def __init__(self, turn_repository: TurnRepository, rawlog_service: RawLogService) -> None:
        self.turn_repository = turn_repository
        self.rawlog_service = rawlog_service

    def create_from_pair(self, user_rawlog: RawLog, assistant_rawlog: RawLog) -> Turn:
        if user_rawlog.speaker_type != "user":
            raise ValueError("user_rawlog must have speaker_type 'user'")
        if assistant_rawlog.speaker_type != "assistant":
            raise ValueError("assistant_rawlog must have speaker_type 'assistant'")
        if user_rawlog.session_id != assistant_rawlog.session_id:
            raise ValueError("rawlogs must belong to the same session")

        existing = self.turn_repository.get_by_rawlog_range(user_rawlog.rawlog_id, assistant_rawlog.rawlog_id)
        if existing is not None:
            return existing

        turn = Turn(
            turn_id=new_id(),
            session_id=assistant_rawlog.session_id,
            start_rawlog_id=user_rawlog.rawlog_id,
            end_rawlog_id=assistant_rawlog.rawlog_id,
            started_at=user_rawlog.occurred_at,
            ended_at=assistant_rawlog.occurred_at,
            metadata_json={
                "builder": "reply_pair_v1",
                "user_sequence_no": user_rawlog.sequence_no,
                "assistant_sequence_no": assistant_rawlog.sequence_no,
            },
        )
        return self.turn_repository.create(turn)

    def build_from_session(self, session_id: str) -> list[Turn]:
        rawlogs = self.rawlog_service.list_session_rawlogs(session_id)
        rawlog_by_id = {rawlog.rawlog_id: rawlog for rawlog in rawlogs}
        turns: list[Turn] = []

        for index, rawlog in enumerate(rawlogs):
            if rawlog.speaker_type != "assistant":
                continue

            user_rawlog = None
            if rawlog.reply_to_rawlog_id:
                candidate = rawlog_by_id.get(rawlog.reply_to_rawlog_id)
                if candidate is not None and candidate.speaker_type == "user":
                    user_rawlog = candidate

            if user_rawlog is None:
                for previous in reversed(rawlogs[:index]):
                    if previous.speaker_type == "user":
                        user_rawlog = previous
                        break

            if user_rawlog is not None:
                turns.append(self.create_from_pair(user_rawlog, rawlog))

        return turns

    def list_session_turns(self, session_id: str) -> list[Turn]:
        return self.turn_repository.list_by_session_id(session_id)
