from app.ingestion import _table_to_markdown


def test_table_to_markdown_basic():
    rows = [["Item", "Qty", "Price"], ["Widget", "2", "10.00"]]
    md = _table_to_markdown(rows)
    lines = md.splitlines()
    assert lines[0] == "| Item | Qty | Price |"
    assert set(lines[1].replace(" ", "")) == {"|", "-"}
    assert lines[2] == "| Widget | 2 | 10.00 |"


def test_table_handles_none_cells_and_newlines():
    rows = [["A", None], ["multi\nline", "B"]]
    md = _table_to_markdown(rows)
    assert "multi line" in md
    assert "|  |" in md or "| A |  |" in md


def test_table_pads_ragged_rows():
    rows = [["a", "b", "c"], ["x"]]
    md = _table_to_markdown(rows)

    assert md.splitlines()[-1].count("|") == 4


def test_empty_table_returns_empty():
    assert _table_to_markdown([]) == ""
    assert _table_to_markdown([[None, None]]) == ""
