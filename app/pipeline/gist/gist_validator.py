class GistValidator:
    MIN_TEXT_LENGTH = 20

    def is_valid(self, gist_data: dict | None) -> bool:
        if not isinstance(gist_data, dict):
            return False
        title = gist_data.get("title", "")
        gist_text = gist_data.get("gist_text", "")
        if not isinstance(title, str) or not title.strip():
            return False
        if not isinstance(gist_text, str) or len(gist_text.strip()) < self.MIN_TEXT_LENGTH:
            return False
        return True
