import os
from pathlib import Path
from typing import Optional, Union

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

    def write_file(
        self,
        container: str,
        path: str,
        data: Union[str, bytes],
        overwrite: bool = True,
    ) -> None:
        """Write a file to the data lake.
        Args:
            container: The container to write the file to.
            path: The path to write the file to.
            data: The data to write to the file.
            overwrite: Whether to overwrite the file if it already exists.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        file_client = self._client.get_file_system_client(container).get_file_client(path)
        file_client.upload_data(data, overwrite=overwrite)

    def upload_file(
        self,
        container: str,
        path: str,
        local_path: str,
        overwrite: bool = True,
    ) -> None:
        file_client = self._client.get_file_system_client(container).get_file_client(path)
        with open(local_path, "rb") as f:
            file_client.upload_data(f, overwrite=overwrite)
    
    def list_files(self, container: str, path: str) -> list:
        """List the files in the data lake.
        Args:
            container: The container to list the files from.
            path: The path to list the files from.
        """
        file_system_client = self._client.get_file_system_client(container)
        return list(file_system_client.get_paths(path)._page_iterator)
    
    def read_file(self, container: str, path: str) -> str:
        """Read a file from the data lake.
        Args:
            container: The container to read the file from.
            path: The path to read the file from.
        """
        file_client = self._client.get_file_system_client(container).get_file_client(path)
        return file_client.download_file().content_as_text()
