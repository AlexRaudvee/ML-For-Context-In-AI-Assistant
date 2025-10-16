from __future__ import annotations
from typing import Any, Dict, Optional
import requests


def get_json(url: str, timeout: int = 20) -> Dict[str, Any]:
    """
    Fetch JSON from a URL.

    Example
    -------
    >>> data = get_json("https://httpbin.org/json")
    >>> print(list(data.keys()))
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def post_form(url: str, data: Dict[str, Any], timeout: int = 20) -> str:
    """
    POST form-encoded data and return the response text.

    Example
    -------
    >>> html = post_form("https://httpbin.org/post", {"name": "alice"})
    """
    resp = requests.post(url, data=data, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def download_file(url: str, dest_path: str, chunk_size: int = 8192, timeout: int = 60) -> None:
    """
    Stream-download a file to disk.

    Example
    -------
    >>> download_file("https://httpbin.org/image/png", "logo.png")
    """
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
