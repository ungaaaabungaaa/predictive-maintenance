"""Strict FD001 readers and a checksum-pinned, NASA-linked data downloader."""

import time
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

DATA_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
ARCHIVE_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
INNER_SHA256 = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"
FILE_SHA256 = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
}
COLUMNS = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]
MAX_DOWNLOAD_BYTES = 30_000_000


def digest(path: Path) -> str:
    """Hash bounded input files used in an experiment."""
    if path.stat().st_size > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"File exceeds 30 MB limit: {path.name}")
    return sha256(path.read_bytes()).hexdigest()


def read_trajectories(path: Path) -> pd.DataFrame:
    """Read finite, complete trajectories with integer IDs and consecutive cycles."""
    digest(path)
    values = np.loadtxt(path, ndmin=2)
    if values.shape[1] != 26 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Trajectory input must have 26 finite columns and at least one row.")
    ids = values[:, :2]
    if (ids < 1).any() or not np.equal(ids, np.floor(ids)).all():
        raise ValueError("Unit IDs and cycles must be positive integers.")
    frame = pd.DataFrame(values, columns=COLUMNS)
    frame[["unit", "cycle"]] = frame[["unit", "cycle"]].astype(int)
    frame = frame.sort_values(["unit", "cycle"], ignore_index=True)
    units = frame.unit.unique()
    if not np.array_equal(units, np.arange(1, len(units) + 1)):
        raise ValueError("Unit IDs must be consecutive, starting at 1.")
    if not np.array_equal(frame.cycle, frame.groupby("unit").cumcount() + 1):
        raise ValueError("Each unit must have consecutive cycles starting at 1, without duplicates.")
    return frame


def read_offsets(path: Path, unit_count: int) -> np.ndarray:
    """Read the official final-cycle RUL vector in ascending unit order."""
    digest(path)
    values = np.loadtxt(path, ndmin=2)
    if values.shape != (unit_count, 1) or not np.isfinite(values).all():
        raise ValueError("RUL input must have one finite value per test unit.")
    result = values[:, 0]
    if (result < 0).any() or not np.equal(result, np.floor(result)).all():
        raise ValueError("RUL offsets must be nonnegative integer cycles.")
    return result


def extract_fd001(payload: bytes, destination: Path) -> dict[str, str]:
    """Verify the archive and copy only three named files, never archive paths."""
    checksum = sha256(payload).hexdigest()
    if checksum not in (ARCHIVE_SHA256, INNER_SHA256):
        raise ValueError("Dataset archive checksum does not match the pinned NASA release.")
    try:
        if checksum == ARCHIVE_SHA256:
            with ZipFile(BytesIO(payload)) as outer:
                payload = outer.read(
                    "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip"
                )
            if sha256(payload).hexdigest() != INNER_SHA256:
                raise ValueError("Inner dataset archive checksum does not match.")
        with ZipFile(BytesIO(payload)) as archive:
            files = {name: archive.read(name) for name in FILE_SHA256}
    except (BadZipFile, KeyError) as exc:
        raise ValueError("Dataset archive does not contain the expected FD001 files.") from exc
    if any(sha256(content).hexdigest() != FILE_SHA256[name] for name, content in files.items()):
        raise ValueError("Dataset member checksum does not match.")
    destination.mkdir(parents=True, exist_ok=True)
    for name in files:
        path = destination / name
        if path.exists() and digest(path) != FILE_SHA256[name]:
            raise ValueError(f"Refusing to replace different data: {path}")
    created = []
    try:
        for name, content in files.items():
            path = destination / name
            if not path.exists():
                with path.open("xb") as stream:
                    created.append(path)
                    stream.write(content)
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return FILE_SHA256.copy()


def download_fd001(destination: Path) -> dict[str, str]:
    """Explicit network operation; reuse only exact verified local files."""
    if all((destination / name).is_file() for name in FILE_SHA256):
        actual = {name: digest(destination / name) for name in FILE_SHA256}
        if actual != FILE_SHA256:
            raise ValueError("Local FD001 files differ from the pinned release. Use a new folder.")
        return actual
    started = time.monotonic()
    chunks, size = [], 0
    with urlopen(DATA_URL, timeout=30) as response:
        while content := response.read(256 * 1024):
            size += len(content)
            if size > MAX_DOWNLOAD_BYTES or time.monotonic() - started > 120:
                raise ValueError("Dataset download exceeded its byte or time limit.")
            chunks.append(content)
    return extract_fd001(b"".join(chunks), destination)
