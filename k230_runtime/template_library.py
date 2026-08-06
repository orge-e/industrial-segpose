"""Read-only loader for compact K230 template bundles."""

try:
    import ujson as json
except ImportError:
    import json

try:
    import uhashlib as hashlib
except ImportError:
    import hashlib

try:
    import uos as os
except ImportError:
    import os


def _normalize_path(path):
    return str(path).replace("\\", "/")


def _dirname(path):
    normalized = _normalize_path(path).rstrip("/")
    separator = normalized.rfind("/")
    if separator < 0:
        return ""
    if separator == 0:
        return "/"
    return normalized[:separator]


def _join(root, child):
    child = _normalize_path(child)
    if child.startswith("/"):
        return child
    root = _normalize_path(root).rstrip("/")
    if not root:
        return child
    return root + "/" + child


class K230TemplateLibrary:
    FORMAT_VERSION = 1

    def __init__(self, manifest_path):
        self.manifest_path = manifest_path
        # CanMV's ``uos`` module intentionally omits ``os.path``.
        self.root = _dirname(manifest_path) or "."
        self.manifest = None
        self.templates = []

    def _read_json(self, path):
        # Keep this compatible with CanMV MicroPython's reduced open().
        with open(path, "r") as stream:
            return json.loads(stream.read())

    def _write_json_atomic(self, path, payload):
        temporary = path + ".tmp"
        with open(temporary, "w") as stream:
            stream.write(json.dumps(payload))
        try:
            os.remove(path)
        except OSError:
            pass
        os.rename(temporary, path)

    def _digest(self, path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                digest.update(chunk)
        value = digest.digest()
        return "".join("%02x" % byte for byte in value)

    def load(self, verify_assets=True):
        manifest = self._read_json(self.manifest_path)
        if manifest.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("unsupported K230 template bundle version")
        loaded = []
        for summary in manifest.get("templates", []):
            metadata_path = _join(self.root, summary["metadata_file"])
            if verify_assets and self._digest(metadata_path) != summary["metadata_sha256"]:
                raise ValueError("template metadata checksum mismatch: %s" % summary["template_id"])
            metadata = self._read_json(metadata_path)
            if metadata.get("template_id") != summary.get("template_id"):
                raise ValueError("template identity mismatch")
            # Older or manually copied metadata can omit a display name even
            # though the manifest summary still contains it.  Keep template
            # identity visible in the detection UI in either case.
            metadata["name"] = metadata.get("name") or summary.get("name") or metadata["template_id"]
            template_root = _dirname(metadata_path)
            metadata["asset_paths"] = {}
            for key, filename in metadata.get("assets", {}).items():
                asset_path = _join(template_root, filename)
                if verify_assets and self._digest(asset_path) != metadata["sha256"][key]:
                    raise ValueError("template asset checksum mismatch: %s/%s" % (summary["template_id"], key))
                metadata["asset_paths"][key] = asset_path
            loaded.append(metadata)
        self.manifest = manifest
        self.templates = loaded
        return self

    def get(self, template_id):
        for template in self.templates:
            if template["template_id"] == template_id:
                return template
        raise KeyError(template_id)

    def set_enabled(self, template_id, enabled):
        if self.manifest is None:
            self.load(verify_assets=False)
        for summary in self.manifest.get("templates", []):
            if summary.get("template_id") != template_id:
                continue
            metadata_path = _join(self.root, summary["metadata_file"])
            metadata = self._read_json(metadata_path)
            metadata["enabled"] = bool(enabled)
            summary["enabled"] = bool(enabled)
            self._write_json_atomic(metadata_path, metadata)
            self._write_json_atomic(self.manifest_path, self.manifest)
            self.load(verify_assets=False)
            return
        raise KeyError(template_id)

    def remove(self, template_id):
        if self.manifest is None:
            self.load(verify_assets=False)
        original = self.manifest.get("templates", [])
        retained = [item for item in original if item.get("template_id") != template_id]
        if len(retained) == len(original):
            raise KeyError(template_id)
        self.manifest["templates"] = retained
        self._write_json_atomic(self.manifest_path, self.manifest)
        self.load(verify_assets=False)
