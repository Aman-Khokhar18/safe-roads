from pathlib import Path
from urllib.parse import urlsplit
import requests

CHUNK_SIZE = 1024 * 64
DEFAULT_TIMEOUT = 60

def download_file(
    url: str,
    outdir: str | Path,
    filename: str | None = None,
    overwrite: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> Path:
    """
    Download `url` to `outdir`.
    - If `filename` is None, infer from the URL path.
    - Set `overwrite=True` to replace existing files.

    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        name = Path(urlsplit(url).path).name or "download.bin"
    else:
        name = filename

    dest = outdir / name
    if dest.exists() and not overwrite:
        return dest

    try:
        with requests.get(url, stream=True, allow_redirects=True, timeout=timeout) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        return dest

    except requests.exceptions.Timeout as e:
        # Clean up partial file if created
        try: dest.unlink(missing_ok=True)
        except Exception: pass
        raise TimeoutError(f"Timed out downloading {url} after {timeout}s") from e

    except requests.exceptions.InvalidURL as e:
        try: dest.unlink(missing_ok=True)
        except Exception: pass
        raise ValueError(f"Invalid URL: {url}") from e

    except requests.exceptions.HTTPError as e:
        try: dest.unlink(missing_ok=True)
        except Exception: pass
        status = getattr(e.response, "status_code", "unknown")
        raise RuntimeError(f"HTTP {status} while fetching {url}") from e

    except requests.exceptions.ConnectionError as e:
        try: dest.unlink(missing_ok=True)
        except Exception: pass
        raise RuntimeError(f"Connection error while fetching {url}") from e

    except requests.exceptions.RequestException as e:
        # Any other requests-related error
        try: dest.unlink(missing_ok=True)
        except Exception: pass
        raise RuntimeError(f"Request failed for {url}: {e}") from e

    except OSError as e:
        # Filesystem errors (permissions, disk full, etc.)
        try: dest.unlink(missing_ok=True)
        except Exception: pass
        raise OSError(f"Failed to write to {dest}: {e}") from e
