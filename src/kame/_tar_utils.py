import os
import tarfile
from pathlib import Path, PurePosixPath


def _validated_members(tar: tarfile.TarFile, destination: Path) -> list[tarfile.TarInfo]:
    destination = destination.resolve()
    validated: list[tarfile.TarInfo] = []
    for member in tar.getmembers():
        member_path = PurePosixPath(member.name)
        if member_path.is_absolute():
            raise RuntimeError(f"Refusing to extract absolute path from archive: {member.name}")
        if any(part == ".." for part in member_path.parts):
            raise RuntimeError(f"Refusing to extract parent-traversing path from archive: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Refusing to extract linked archive member: {member.name}")
        if member.ischr() or member.isblk() or member.isfifo():
            raise RuntimeError(f"Refusing to extract special archive member: {member.name}")
        if not (member.isdir() or member.isreg()):
            raise RuntimeError(f"Refusing to extract unsupported archive member: {member.name}")

        path_parts = [part for part in member_path.parts if part not in ("", ".")]
        if not path_parts:
            raise RuntimeError(f"Refusing to extract ambiguous archive member: {member.name}")

        target_path = (destination / Path(*path_parts)).resolve(strict=False)
        if os.path.commonpath([str(destination), str(target_path)]) != str(destination):
            raise RuntimeError(f"Refusing to extract archive member outside destination: {member.name}")
        validated.append(member)
    return validated


def extract_data_archive(tar: tarfile.TarFile, destination: Path) -> None:
    validated_members = _validated_members(tar, destination)
    # Use tarfile's data filter when available; older supported Python versions do not provide it.
    if hasattr(tarfile, "data_filter"):
        tar.extractall(path=destination, members=validated_members, filter="data")
        return

    tar.extractall(path=destination, members=validated_members)
