from citeguard.ingest.chunk import chunk_markdown, count_tokens, iter_sections


def test_heading_sections():
    md = "# A\n\nhello\n\n## B\n\nworld\n"
    secs = list(iter_sections(md))
    assert secs[0][0] == "A"
    assert "hello" in secs[0][1]
    assert secs[1][0] == "A > B"


def test_chunk_splits_long_section():
    body = "word " * 2000
    md = f"# Title\n\n{body}"
    chunks = chunk_markdown(
        md,
        doc_id="test:doc.md",
        source_url="https://example.com/doc.md",
        source_repo="example/repo",
        source_commit="abc",
        target_tokens=100,
        overlap_tokens=10,
    )
    assert len(chunks) > 1
    assert all(c.token_count <= 120 for c in chunks)  # slight slack
    assert all(c.doc_id == "test:doc.md" for c in chunks)
    assert all(c.title_path == "Title" for c in chunks)


def test_frontmatter_stripped():
    md = "---\ntitle: x\n---\n\n# H\n\nbody text here\n"
    chunks = chunk_markdown(
        md,
        doc_id="d",
        source_url="u",
        source_repo="r",
        source_commit="c",
    )
    assert chunks
    assert "title: x" not in chunks[0].text


def test_count_tokens_positive():
    assert count_tokens("hello world") > 0


def test_chunk_ids_unique_within_and_across_docs():
    from citeguard.ingest.chunk import chunk_markdown, make_chunk_id

    md = "# A\n\n" + ("word " * 800) + "\n\n# B\n\n" + ("other " * 800)
    a = chunk_markdown(md, doc_id="repo:a.md", source_url="u", source_repo="r", source_commit="c", target_tokens=50, overlap_tokens=5)
    b = chunk_markdown(md, doc_id="repo:b.md", source_url="u", source_repo="r", source_commit="c", target_tokens=50, overlap_tokens=5)
    ids = [c.chunk_id for c in a] + [c.chunk_id for c in b]
    assert len(ids) == len(set(ids))
    assert a[0].chunk_id == make_chunk_id("repo:a.md", 0)
    assert a[1].chunk_id == make_chunk_id("repo:a.md", 1)


def test_hash_comments_in_code_fences_are_not_headings():
    from citeguard.ingest.chunk import iter_sections

    md = "# Title\n\nText.\n\n```python\n# a comment\nx = 1\n```\n\nMore."
    secs = list(iter_sections(md))
    assert [t for t, _ in secs] == ["Title"]


def test_resolve_includes(tmp_path):
    from citeguard.ingest.pipeline import resolve_includes

    (tmp_path / "docs_src").mkdir()
    (tmp_path / "docs_src" / "a.py").write_text("raise HTTPException(status_code=404)\n")
    base = tmp_path / "docs" / "en"
    base.mkdir(parents=True)
    out, ok, miss = resolve_includes(
        'X\n{* ../../docs_src/a.py hl[1] title["a.py"] *}\n{* ../../docs_src/none.py *}', base
    )
    assert ok == 1 and miss == 1 and "raise HTTPException" in out and "```python" in out
