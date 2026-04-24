import io
import tarfile
from pathlib import Path

import pytest

from kame._tar_utils import extract_data_archive


def _write_tar_member(tar: tarfile.TarFile, name: str, data: bytes = b"") -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    tar.addfile(member, io.BytesIO(data))


def test_extract_data_archive_allows_regular_files(tmp_path: Path) -> None:
    archive_path = tmp_path / "dist.tgz"
    destination = tmp_path / "extract"
    destination.mkdir()
    with tarfile.open(archive_path, "w:gz") as tar:
        _write_tar_member(tar, "dist/index.html", b"hello")

    with tarfile.open(archive_path, "r:gz") as tar:
        extract_data_archive(tar, destination)

    assert (destination / "dist" / "index.html").read_text() == "hello"


def test_extract_data_archive_rejects_parent_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "dist.tgz"
    destination = tmp_path / "extract"
    destination.mkdir()
    with tarfile.open(archive_path, "w:gz") as tar:
        _write_tar_member(tar, "../escape.txt", b"nope")

    with tarfile.open(archive_path, "r:gz") as tar:
        with pytest.raises(RuntimeError, match="parent-traversing"):
            extract_data_archive(tar, destination)


def test_extract_data_archive_rejects_symlinks(tmp_path: Path) -> None:
    archive_path = tmp_path / "dist.tgz"
    destination = tmp_path / "extract"
    destination.mkdir()
    with tarfile.open(archive_path, "w:gz") as tar:
        member = tarfile.TarInfo("dist/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "index.html"
        tar.addfile(member)

    with tarfile.open(archive_path, "r:gz") as tar:
        with pytest.raises(RuntimeError, match="linked archive member"):
            extract_data_archive(tar, destination)
