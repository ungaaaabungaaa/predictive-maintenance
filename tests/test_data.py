from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from predictive_maintenance import data
from predictive_maintenance.data import extract_fd001, read_offsets, read_trajectories


def make_rows(units=2, cycles=3):
    return np.array([[u, c, *([float(c)] * 24)]
                     for u in range(1, units + 1) for c in range(1, cycles + 1)])


def save_rows(path: Path, rows):
    np.savetxt(path, rows)
    return path


def test_loads_sorted_complete_trajectories(tmp_path):
    frame = read_trajectories(save_rows(tmp_path / "rows.txt", make_rows()))
    assert list(frame.columns) == ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [
        f"sensor_{i}" for i in range(1, 22)]
    assert len(frame) == 6
    assert list(frame.groupby("unit").size()) == [3, 3]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nan", "columns", "unit", "fractional"])
def test_rejects_invalid_trajectories(tmp_path, mutation):
    rows = make_rows()
    if mutation == "missing":
        rows = np.delete(rows, 1, axis=0)
    elif mutation == "duplicate":
        rows = np.vstack([rows, rows[0]])
    elif mutation == "nan":
        rows[0, 6] = np.nan
    elif mutation == "columns":
        rows = rows[:, :-1]
    elif mutation == "unit":
        rows[3:, 0] = 3
    else:
        rows[0, 1] = 1.5
    with pytest.raises(ValueError):
        read_trajectories(save_rows(tmp_path / "rows.txt", rows))


def test_offsets_have_one_finite_nonnegative_value_per_unit(tmp_path):
    path = save_rows(tmp_path / "rul.txt", np.array([12, 30]))
    np.testing.assert_array_equal(read_offsets(path, 2), [12, 30])
    for values in [[12], [12, -1], [12, np.nan], [12, 1.5], [[1, 2], [3, 4]]]:
        save_rows(path, np.array(values))
        with pytest.raises(ValueError):
            read_offsets(path, 2)


def test_extracts_only_expected_members_and_rejects_untrusted_bytes(tmp_path):
    contents = BytesIO()
    with ZipFile(contents, "w") as archive:
        archive.writestr("../../outside.txt", "unsafe")
        for name in ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]:
            archive.writestr(name, "data")
    with pytest.raises(ValueError, match="checksum"):
        extract_fd001(contents.getvalue(), tmp_path)
    assert not list(tmp_path.iterdir())


def pinned_archive(monkeypatch):
    files = {name: name.encode() for name in data.FILE_SHA256}
    inner = BytesIO()
    with ZipFile(inner, "w") as archive:
        archive.writestr("../../outside.txt", "unsafe")
        for name, content in files.items():
            archive.writestr(name, content)
    outer = BytesIO()
    with ZipFile(outer, "w") as archive:
        archive.writestr("6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip",
                         inner.getvalue())
    monkeypatch.setattr(data, "ARCHIVE_SHA256", sha256(outer.getvalue()).hexdigest())
    monkeypatch.setattr(data, "INNER_SHA256", sha256(inner.getvalue()).hexdigest())
    monkeypatch.setattr(data, "FILE_SHA256", {n: sha256(c).hexdigest() for n, c in files.items()})
    return outer.getvalue(), inner.getvalue()


def test_verified_nested_or_inner_archive_extracts_only_allowlisted_files(tmp_path, monkeypatch):
    outer, inner = pinned_archive(monkeypatch)
    assert data.extract_fd001(outer, tmp_path) == data.FILE_SHA256
    assert data.extract_fd001(inner, tmp_path) == data.FILE_SHA256
    assert {p.name for p in tmp_path.iterdir()} == set(data.FILE_SHA256)
    assert not (tmp_path.parent / "outside.txt").exists()
    (tmp_path / "RUL_FD001.txt").write_text("different")
    with pytest.raises(ValueError, match="Refusing to replace"):
        data.extract_fd001(outer, tmp_path)


def test_downloader_pins_source_and_reuses_only_verified_local_files(tmp_path, monkeypatch):
    outer, _ = pinned_archive(monkeypatch)
    calls = []

    def open_source(url, timeout):
        calls.append((url, timeout))
        return BytesIO(outer)

    monkeypatch.setattr(data, "urlopen", open_source)
    assert data.download_fd001(tmp_path) == data.FILE_SHA256
    assert calls == [(data.DATA_URL, 30)]
    assert data.download_fd001(tmp_path) == data.FILE_SHA256
    assert len(calls) == 1
    (tmp_path / "RUL_FD001.txt").write_text("tampered")
    with pytest.raises(ValueError, match="differ"):
        data.download_fd001(tmp_path)


def test_download_size_limit_stops_before_any_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: BytesIO(b"12345"))
    monkeypatch.setattr(data, "MAX_DOWNLOAD_BYTES", 4)
    with pytest.raises(ValueError, match="limit"):
        data.download_fd001(tmp_path)
    assert not list(tmp_path.iterdir())


def test_dataset_member_integrity_and_input_size_bounds(tmp_path, monkeypatch):
    outer, _ = pinned_archive(monkeypatch)
    monkeypatch.setattr(data, "FILE_SHA256", dict.fromkeys(data.FILE_SHA256, "wrong"))
    with pytest.raises(ValueError, match="member checksum"):
        data.extract_fd001(outer, tmp_path)
    path = tmp_path / "large.txt"
    path.write_text("12345")
    monkeypatch.setattr(data, "MAX_DOWNLOAD_BYTES", 4)
    with pytest.raises(ValueError, match="limit"):
        data.digest(path)
