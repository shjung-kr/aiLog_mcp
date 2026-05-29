"""
aiLog MCP Server

Claude Desktop / ChatGPT 등 MCP 클라이언트가 aiLog의 기억 시스템을 도구로 사용할 수 있게 합니다.

사용법:
  python server.py        # stdio transport (Claude Desktop 등)
  python server.py sse    # SSE transport (HTTP 기반 클라이언트)

환경변수: .env 파일 또는 시스템 환경변수로 설정 (예: .env.example 참조)
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from mcp.server.fastmcp import FastMCP

from app.db.repositories.episode_repository import EpisodeRepository
from app.db.repositories.long_term_memory_repository import LongTermMemoryRepository
from app.db.repositories.rawlog_repository import RawLogRepository
from app.db.repositories.search_repository import SearchRepository
from app.db.repositories.session_repository import SessionRepository
from app.db.session import SessionLocal
from app.llm.client import LLMClient
from app.services.episode_idle_scheduler import episode_idle_scheduler
from app.services.rawlog_service import RawLogService
from app.services.retrieval_service import RetrievalService
from app.services.session_service import SessionService
from app.utils.datetime import utc_now

mcp = FastMCP(
    "aiLog",
    instructions=(
        "당신은 aiLog 메모리 시스템에 연결된 AI입니다. "
        "대화를 시작할 때 반드시 다음 순서를 따르세요:\n"
        "1. get_long_term_memories 를 호출해 사용자의 장기기억(프로필, 관심사, 스타일)을 파악합니다.\n"
        "2. create_session 으로 새 세션을 만들고 session_id를 기억합니다.\n"
        "3. 매 메시지마다 log_turn 을 호출해 user/assistant 발화를 저장합니다.\n"
        "4. 사용자가 과거 대화나 기억을 언급하면 search_memories 를 호출해 관련 맥락을 가져옵니다.\n"
        "5. 검색된 맥락이 있으면 자연스럽게 대화에 반영하되, '기억합니다' 등의 메타 표현은 피합니다.\n"
        "6. 검색된 기억이 없으면 솔직하게 기록이 없다고 말하고, 일반 지식으로 대체하지 마세요."
    ),
)


# ── 세션 관리 ──────────────────────────────────────────────────────────────────

@mcp.tool()
def create_session(title: str | None = None) -> dict:
    """
    새 대화 세션을 생성합니다.

    대화를 시작할 때 한 번 호출하세요. 반환된 session_id를 이후 모든 도구 호출에 사용합니다.

    Args:
        title: 세션 제목 (선택). 대화 주제를 간략히 적으면 나중에 구별하기 좋습니다.

    Returns:
        session_id, title, started_at
    """
    db = SessionLocal()
    try:
        service = SessionService(SessionRepository(db))
        session = service.create_session(title=title)
        db.commit()
        return {
            "session_id": session.session_id,
            "title": session.title,
            "started_at": session.started_at.isoformat(),
        }
    finally:
        db.close()


@mcp.tool()
def list_sessions(limit: int = 20) -> list[dict]:
    """
    최근 대화 세션 목록을 반환합니다.

    이전 대화를 이어가거나 과거 세션 ID가 필요할 때 사용하세요.

    Args:
        limit: 최대 반환 개수 (기본 20)
    """
    db = SessionLocal()
    try:
        service = SessionService(SessionRepository(db))
        sessions = service.list_sessions(limit=limit)
        return [
            {
                "session_id": s.session_id,
                "title": s.title,
                "started_at": s.started_at.isoformat(),
                "last_activity_at": s.last_activity_at.isoformat(),
                "status": s.status,
            }
            for s in sessions
        ]
    finally:
        db.close()


# ── 대화 저장 ──────────────────────────────────────────────────────────────────

@mcp.tool()
def log_turn(session_id: str, role: str, content: str, model: str | None = None) -> dict:
    """
    대화 1턴(메시지 1개)을 aiLog에 저장합니다.

    user 발화와 assistant 발화 모두 매번 저장해야 합니다.
    저장 후 백그라운드에서 에피소드/기억 생성 파이프라인이 자동 실행됩니다.

    Args:
        session_id: create_session 에서 받은 세션 ID
        role: 'user' 또는 'assistant'
        content: 메시지 전체 내용
        model: 어시스턴트 모델명 (선택, 예: 'claude-3-5-sonnet-20241022')

    Returns:
        rawlog_id, sequence_no, speaker_type
    """
    if role not in ("user", "assistant"):
        return {"error": f"role은 'user' 또는 'assistant' 여야 합니다. (받은 값: {role!r})"}

    db = SessionLocal()
    try:
        session_service = SessionService(SessionRepository(db))
        rawlog_service = RawLogService(RawLogRepository(db), session_service)

        seq = rawlog_service.get_next_sequence_no(session_id)
        rawlog = rawlog_service.create_rawlog(
            session_id=session_id,
            sequence_no=seq,
            speaker_type=role,
            content=content,
            occurred_at=utc_now(),
            source_model=model,
        )
        db.commit()

        # 마지막 메시지로부터 EPISODE_IDLE_SECONDS 후 에피소드 빌드 예약
        episode_idle_scheduler.schedule(session_id)

        return {
            "rawlog_id": rawlog.rawlog_id,
            "sequence_no": rawlog.sequence_no,
            "speaker_type": rawlog.speaker_type,
        }
    except LookupError as exc:
        db.rollback()
        return {"error": str(exc)}
    finally:
        db.close()


@mcp.tool()
def get_recent_history(session_id: str, limit: int = 20) -> list[dict]:
    """
    세션의 최근 대화 기록을 반환합니다.

    현재 대화의 앞 맥락을 참고하거나, 같은 세션을 이어갈 때 사용하세요.

    Args:
        session_id: 세션 ID
        limit: 가져올 최근 턴 수 (기본 20)
    """
    db = SessionLocal()
    try:
        rawlog_service = RawLogService(RawLogRepository(db), SessionService(SessionRepository(db)))
        rawlogs = rawlog_service.list_session_rawlogs(session_id)
        recent = rawlogs[-limit:] if len(rawlogs) > limit else rawlogs
        return [
            {
                "role": r.speaker_type,
                "content": r.content,
                "occurred_at": r.occurred_at.isoformat(),
            }
            for r in recent
        ]
    except LookupError:
        return []
    finally:
        db.close()


# ── 기억 검색 / 조회 ───────────────────────────────────────────────────────────

@mcp.tool()
def search_memories(query: str, session_id: str | None = None) -> dict:
    """
    과거 대화에서 관련 기억(에피소드)을 의미 기반으로 검색합니다.

    사용자가 "저번에 얘기했던 거", "기억해?", 특정 과거 주제를 언급할 때 호출하세요.
    검색 결과가 있으면 context 필드의 내용을 답변에 자연스럽게 반영하세요.

    Args:
        query: 검색할 내용 (사용자 발화 그대로 넣어도 됩니다)
        session_id: 현재 세션 ID (있으면 최근 대화 맥락도 검색에 반영됩니다)

    Returns:
        found(bool), context(str), episodes(list)
    """
    db = SessionLocal()
    try:
        rawlog_service = RawLogService(RawLogRepository(db), SessionService(SessionRepository(db)))
        retrieval = RetrievalService(
            episode_repository=EpisodeRepository(db),
            rawlog_service=rawlog_service,
            llm_client=LLMClient(),
            search_repository=SearchRepository(db),
        )

        recent_turns: list[str] | None = None
        if session_id:
            try:
                rawlogs = rawlog_service.list_session_rawlogs(session_id)
                recent_turns = [f"{r.speaker_type}: {r.content}" for r in rawlogs[-6:]]
            except LookupError:
                pass

        context_text, context_items = retrieval.retrieve_for_query(
            query, session_id=session_id, recent_turns=recent_turns
        )
        db.commit()  # search_log 저장

        return {
            "found": context_text is not None,
            "context": context_text or "",
            "episodes": context_items,
        }
    finally:
        db.close()


@mcp.tool()
def get_long_term_memories(limit: int = 50) -> list[dict]:
    """
    저장된 장기기억 전체를 반환합니다.

    대화 시작 시 호출해 사용자의 프로필, 관심사, 말투 스타일 등을 파악하세요.
    중요도(importance) 순으로 정렬됩니다.

    Args:
        limit: 최대 반환 개수 (기본 50)
    """
    db = SessionLocal()
    try:
        ltm_repo = LongTermMemoryRepository(db)
        memories = ltm_repo.list_all(limit=limit)
        return [
            {
                "memory_id": m.memory_id,
                "title": m.title,
                "type": m.memory_type,
                "text": m.memory_text,
                "importance": m.importance_score,
            }
            for m in memories
        ]
    finally:
        db.close()


# ── 진입점 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
