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
