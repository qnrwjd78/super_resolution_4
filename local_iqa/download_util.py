import os
from urllib.parse import urlparse

from torch.hub import download_url_to_file, get_dir


DEFAULT_CACHE_DIR = os.path.join(get_dir(), "local_iqa")


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Download a remote checkpoint into the local cache if needed."""
    model_dir = model_dir or DEFAULT_CACHE_DIR
    os.makedirs(model_dir, exist_ok=True)

    filename = os.path.basename(urlparse(url).path)
    if file_name is not None:
        filename = file_name

    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file
