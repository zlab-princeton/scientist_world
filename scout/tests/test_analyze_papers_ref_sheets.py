import analyze_papers


def test_snapshot_counts_rows_not_blank_lines(tmp_path):
    sheets = analyze_papers.REF_SHEETS
    (tmp_path / sheets[0]).write_text("")
    (tmp_path / sheets[1]).write_text('{"name": "a"}\n\n{"name": "b"}\n')

    counts = analyze_papers._snapshot_ref_sheets(tmp_path)
    assert counts[sheets[0]] == 0
    assert counts[sheets[1]] == 2
    assert counts[sheets[2]] == 0                      # missing file
