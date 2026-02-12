from pathlib import Path

from domain.models import StoredObject


class CloudflareR2StorageGateway:
    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        public_base_url: str | None = None,
        signed_url_expiration_seconds: int = 3600,
    ) -> None:
        import boto3

        self._bucket_name = bucket_name
        self._public_base_url = public_base_url.rstrip("/") if public_base_url else None
        self._signed_url_expiration_seconds = signed_url_expiration_seconds
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
        )

    def upload_file(
        self,
        source_path: Path,
        object_key: str,
        content_type: str | None = None,
    ) -> StoredObject:
        if not source_path.exists():
            raise FileNotFoundError(f"Missing: {source_path}")

        put_kwargs: dict[str, str] = {}
        if content_type:
            put_kwargs["ContentType"] = content_type

        with source_path.open("rb") as file_handle:
            self._client.put_object(
                Bucket=self._bucket_name,
                Key=object_key,
                Body=file_handle,
                **put_kwargs,
            )

        object_url = (
            f"{self._public_base_url}/{object_key}"
            if self._public_base_url
            else f"s3://{self._bucket_name}/{object_key}"
        )
        signed_url = self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket_name, "Key": object_key},
            ExpiresIn=self._signed_url_expiration_seconds,
        )
        return StoredObject(object_key=object_key, object_url=object_url, signed_url=signed_url)
