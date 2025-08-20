import os
import io
import json
import builtins
import tempfile
import pytest

from src.rag_tools import add_source


class DummyResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


def test_get_clean_filename_from_url_path():
    assert add_source.get_clean_filename_from_url_path("/abc/def") == "abc_def"
    assert add_source.get_clean_filename_from_url_path("/") == "index"


def test_download_html_success(monkeypatch, tmp_path):
    def fake_get(url, headers=None):
        return DummyResponse("<html><body><p>ciao</p></body></html>")

    monkeypatch.setattr(add_source.requests, "get", fake_get)

    fn, path = add_source.download_html("http://test.com/page", output_dir=tmp_path)

    assert fn.endswith(".html")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        assert "ciao" in f.read()


def test_download_html_failure(monkeypatch):
    def fake_get(url, headers=None):
        return DummyResponse("fail", status_code=404)

    monkeypatch.setattr(add_source.requests, "get", fake_get)

    fn, path = add_source.download_html("http://fail.com/page")
    assert fn is None
    assert path is None


def test_load_and_save_source_index(tmp_path, monkeypatch):
    index_path = tmp_path / "fonte_index.json"
    monkeypatch.setattr(add_source, "SOURCE_INDEX_PATH", str(index_path))

    # inizialmente vuoto
    assert add_source.load_source_index() == []

    sample = [{"id": "abc"}]
    add_source.save_source_index(sample)

    with open(index_path, encoding="utf-8") as f:
        content = json.load(f)

    assert content == sample
    assert add_source.load_source_index() == sample


def test_register_source_metadata(monkeypatch, tmp_path):
    index_path = tmp_path / "fonte_index.json"
    monkeypatch.setattr(add_source, "SOURCE_INDEX_PATH", str(index_path))

    # mock input()
    inputs = iter(["Titolo", "storia", "sec1", "2"])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    add_source.register_source_metadata("file.html", "http://url.com")

    with open(index_path, encoding="utf-8") as f:
        content = json.load(f)

    assert content[0]["titolo"] == "Titolo"
    assert content[0]["materia"] == "storia"
    assert content[0]["classe"] == "sec1"
    assert content[0]["anno"] == 2
    assert content[0]["fonte"] == "http://url.com"


def test_extract_text_from_html(tmp_path):
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True)
    html_file = raw_dir / "page.html"
    html_file.write_text("<html><body><h1>Titolo</h1><p>Paragrafo</p></body></html>", encoding="utf-8")

    path = add_source.extract_text_from_html(str(html_file))

    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    assert "Titolo" in txt
    assert "Paragrafo" in txt
