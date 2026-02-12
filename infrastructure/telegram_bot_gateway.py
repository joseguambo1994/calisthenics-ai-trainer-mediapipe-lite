from pathlib import Path

import requests


class TelegramBotGateway:
    def __init__(self, bot_token: str, timeout_seconds: int = 120) -> None:
        self._bot_token = bot_token
        self._timeout_seconds = timeout_seconds
        self._api_url = f"https://api.telegram.org/bot{self._bot_token}"
        self._file_url = f"https://api.telegram.org/file/bot{self._bot_token}"

    def download_video(self, file_id: str, target_dir: Path) -> Path:
        file_path = self._resolve_file_path(file_id=file_id)
        filename = Path(file_path).name or f"{file_id}.mp4"

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename

        download_url = f"{self._file_url}/{file_path}"
        with requests.get(download_url, stream=True, timeout=self._timeout_seconds) as response:
            response.raise_for_status()
            with target_path.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_handle.write(chunk)

        return target_path

    def _resolve_file_path(self, file_id: str) -> str:
        response = requests.get(
            f"{self._api_url}/getFile",
            params={"file_id": file_id},
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()

        payload = response.json()
        if not payload.get("ok"):
            description = payload.get("description", "Unknown Telegram API error")
            raise RuntimeError(f"Telegram getFile failed: {description}")

        file_path = payload.get("result", {}).get("file_path")
        if not file_path:
            raise RuntimeError("Telegram getFile response missing result.file_path")

        return file_path
