import os
from typing import Optional

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient


class DataLakeOperator:
    def __init__(
        self,
        account_url: Optional[str] = None,
        connection_string: Optional[str] = None,
        credential: Optional[TokenCredential] = None,
    ):
        conn = connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if conn:
            self._client = DataLakeServiceClient.from_connection_string(conn)
            return

        url = account_url or os.environ["AZURE_STORAGE_ACCOUNT_URL"]
        cred = credential or DefaultAzureCredential()
        self._client = DataLakeServiceClient(account_url=url, credential=cred)

    def list_files(self, container: str, path: str = "") -> list[str]:
        fs = self._client.get_file_system_client(container)
        return [p.name for p in fs.get_paths(path=path)]

    def read_file(self, container: str, path: str, encoding: str = "utf-8") -> str:
        file_client = self._client.get_file_system_client(container).get_file_client(path)
        return file_client.download_file().readall().decode(encoding)
   
