class Segmenter:
    def __init__(self, max_turns_per_segment: int = 5) -> None:
        self.max_turns_per_segment = max_turns_per_segment

    def segment(self, turns: list[dict]) -> list[list[dict]]:
        if not turns:
            return []
        return [
            turns[i : i + self.max_turns_per_segment]
            for i in range(0, len(turns), self.max_turns_per_segment)
        ]
