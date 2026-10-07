"""Acquire official dataset provenance and an archive; keep failures explicit."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import urllib.error
import urllib.parse
import urllib.request

from common import ROOT, inside, now, read_json, sha256, write_json


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def request(url, timeout=45):
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError("Only HTTPS sources are accepted.")
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        "User-Agent": "rff-research/1.0 (public academic dataset acquisition)"}), timeout=timeout)


def small_read(url, path):
    with request(url) as response:
        payload = response.read(2 * 1024 * 1024 + 1)
        if len(payload) > 2 * 1024 * 1024:
            raise ValueError("Metadata response unexpectedly exceeds 2 MiB.")
        path.write_bytes(payload)
        return payload, response.url


def acquire(url, destination, max_bytes, expected_hash=None):
    import tarfile
    import zipfile
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size > max_bytes:
            raise ValueError("Existing archive exceeds configured cap.")
        digest = sha256(destination)
        if expected_hash and digest != expected_hash:
            raise ValueError("Existing archive SHA-256 mismatch.")
        if not zipfile.is_zipfile(destination) and not tarfile.is_tarfile(destination):
            raise ValueError("Existing file is not a readable ZIP/TAR archive.")
        return {"url": url, "path": str(destination), "bytes": destination.stat().st_size,
                "sha256": digest, "reused": True, "publisher_checksum_verified": bool(expected_hash)}
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise ValueError(f"Partial download already exists; inspect it before retrying: {partial}")
    with request(url) as response:
        content_type = response.headers.get("Content-Type", "")
        if "html" in content_type or "json" in content_type:
            raise ValueError(f"Expected an archive, received {content_type}.")
        declared = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None
        if declared and declared > max_bytes:
            raise ValueError(f"Archive exceeds configured cap: {declared} > {max_bytes} bytes.")
        if shutil.disk_usage(destination.parent).free < (declared or max_bytes) * 2:
            raise ValueError("Insufficient free space for download plus extraction.")
        count = 0
        with partial.open("xb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                count += len(chunk)
                if count > max_bytes:
                    raise ValueError("Archive exceeds configured download cap.")
                stream.write(chunk)
        if declared is not None and count != declared:
            raise ValueError("Truncated download.")
    digest = sha256(partial)
    if expected_hash and digest != expected_hash:
        raise ValueError("Downloaded archive SHA-256 mismatch.")
    if not zipfile.is_zipfile(partial) and not tarfile.is_tarfile(partial):
        raise ValueError("Response is not a readable ZIP/TAR archive; .part retained for inspection.")
    partial.rename(destination)
    return {"url": url, "resolved_url": response.url, "path": str(destination), "bytes": count,
            "sha256": digest, "publisher_checksum_verified": bool(expected_hash)}


def extract(archive, destination, max_bytes):
    import tarfile
    import zipfile
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Raw directory must be empty; existing recordings will not be overwritten.")
    kept = []
    total = 0

    def copy_member(name, size, opener):
        nonlocal total
        # Avoid NTFS alternate streams, absolute paths, traversal and archive links.
        normalized = name.replace("\\", "/")
        if ":" in normalized or normalized.startswith("/") or ".." in normalized.split("/"):
            raise ValueError(f"Unsafe archive member: {name}")
        path = inside(destination, normalized)
        if path.suffix.lower() not in [".sigmf-data", ".sigmf-meta", ".bin", ".json"]:
            return
        total += size
        if total > max_bytes:
            raise ValueError("Extracted dataset exceeds configured cap.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with opener() as source, path.open("xb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        if path.stat().st_size != size:
            raise ValueError(f"Truncated archive member: {name}")
        kept.append(normalized)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if member.is_dir():
                    continue
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are unsupported.")
                copy_member(member.filename, member.file_size, lambda m=member: bundle.open(m))
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            for member in bundle:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("Only regular files are accepted in TAR archives.")
                copy_member(member.name, member.size, lambda m=member: bundle.extractfile(m))
    else:
        raise ValueError("Unsupported archive format.")
    if not any(name.endswith((".sigmf-data", ".bin")) for name in kept):
        raise ValueError("Archive contains no waveform recordings.")
    return {"files": len(kept), "extracted_bytes": total, "destination": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["oracle_dataset2", "airid"], default="oracle_dataset2")
    parser.add_argument("--url", help="Explicit archive URL, recorded in provenance; HTTPS only.")
    parser.add_argument("--archive", type=Path, help="Import an already-downloaded public archive.")
    parser.add_argument("--sha256", help="Optional independently obtained archive checksum.")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--max-download-gib", type=float, default=12)
    parser.add_argument("--max-extracted-gib", type=float, default=40)
    args = parser.parse_args()
    if args.max_download_gib <= 0 or args.max_extracted_gib <= 0:
        parser.error("Size caps must be positive.")
    source = read_json(ROOT / "configs/sources.json")[args.dataset]
    receipt_dir = ROOT / "data/receipts" / args.dataset
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {"attempted_at_jst": now(), "dataset": args.dataset, "source": source,
               "status": "running", "attempts": [], "waveform_acquired": False}
    target = receipt_dir / "acquisition.json"
    archive = args.archive
    try:
        discovered = []
        for kind, url in [("landing", source["landing_url"]),
                          ("handle", f'https://hdl.handle.net/api/handles/{source["handle"]}')]:
            try:
                payload, resolved = small_read(url, receipt_dir / f"{kind}.{'json' if kind == 'handle' else 'html'}")
                receipt["attempts"].append({"url": url, "status": "ok", "resolved_url": resolved})
                if kind == "handle":
                    discovered = [v["data"]["value"] for v in json.loads(payload)["values"] if v["type"] == "URL"]
            except (OSError, ValueError) as error:
                receipt["attempts"].append({"url": url, "status": "failed", "error": str(error)})
        record_url = discovered[0] if discovered else source["record_url"]
        candidates = []
        if not archive:
            try:
                payload, resolved = small_read(record_url, receipt_dir / "record.html")
                receipt["attempts"].append({"url": record_url, "status": "ok", "resolved_url": resolved})
                links = Links()
                links.feed(payload.decode("utf-8", errors="replace"))
                candidates = sorted(set(urllib.parse.urljoin(resolved, link) for link in links.hrefs
                                        if "/downloads/" in link))
            except (OSError, ValueError) as error:
                receipt["attempts"].append({"url": record_url, "status": "failed", "error": str(error)})
        receipt["discovered_downloads"] = candidates
        if args.probe_only:
            receipt["status"] = "metadata_only"
        else:
            if archive:
                import tarfile
                import zipfile
                archive = archive.resolve(strict=True)
                if archive.stat().st_size > int(args.max_download_gib * 1024**3):
                    raise ValueError("Imported archive exceeds configured download cap.")
                if not zipfile.is_zipfile(archive) and not tarfile.is_tarfile(archive):
                    raise ValueError("Imported file is not a readable ZIP/TAR archive.")
                digest = sha256(archive)
                if args.sha256 and digest != args.sha256:
                    raise ValueError("Imported archive SHA-256 mismatch.")
                receipt["archive"] = {"path": str(archive), "sha256": digest, "origin": "local_import",
                                      "publisher_checksum_verified": bool(args.sha256)}
            else:
                url = args.url or source.get("archive_url") or (candidates[0] if len(candidates) == 1 else None)
                if not url:
                    raise ValueError("No unique accessible official archive URL. Use --url or --archive after obtaining the public data.")
                archive = ROOT / "data/downloads" / f"{args.dataset}.archive"
                try:
                    receipt["archive"] = acquire(url, archive, int(args.max_download_gib * 1024**3), args.sha256)
                    receipt["attempts"].append({"url": url, "status": "ok", "kind": "archive"})
                except Exception as error:
                    receipt["attempts"].append({"url": url, "status": "failed", "kind": "archive", "error": str(error)})
                    raise
            receipt["waveform_acquired"] = True
            receipt["status"] = "archive_acquired"
            if args.extract:
                receipt["extraction"] = extract(archive, ROOT / "data/raw" / args.dataset,
                                               int(args.max_extracted_gib * 1024**3))
                receipt["status"] = "extracted"
    except Exception as error:
        receipt.update(status="blocked", error=f"{type(error).__name__}: {error}")
        write_json(target, receipt)
        print(f"Acquisition blocked. Receipt: {target}\n{error}")
        raise SystemExit(2)
    write_json(target, receipt)
    print(f"Acquisition: {receipt['status']}; waveform_acquired={receipt['waveform_acquired']}; {target}")


if __name__ == "__main__":
    main()
