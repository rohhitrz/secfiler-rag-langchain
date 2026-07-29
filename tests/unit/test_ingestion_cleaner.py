"""Cleaning contract: XBRL values survive, tables become rows, structure holds.

Fixtures are hand-written miniatures of real filing markup. A unit test should
exercise the *shape* of a problem — a nested inline-XBRL tag, a table with
orphaned currency cells — not the size of an 8 MB document.
"""

import pytest

from secfiler_rag.core.exceptions import IngestionError
from secfiler_rag.ingestion.cleaner import clean_html


def test_script_style_and_head_are_removed():
    html = """
    <head><title>ignore me</title></head>
    <body>
      <script>var x = 'script content';</script>
      <style>.cls { color: red; }</style>
      <p>Real filing prose.</p>
    </body>
    """

    text = clean_html(html)

    assert "Real filing prose." in text
    assert "script content" not in text
    assert "color: red" not in text
    assert "ignore me" not in text


def test_inline_xbrl_values_are_preserved():
    """The regression that motivated this module.

    `ix:nonFraction` wraps the visible number. Removing the tag *with its
    contents* deletes the financial data — which is what the previous build
    did, and why income statements came out as `Products $ $ $`.
    """
    html = """
    <p>Total net sales <ix:nonFraction name="us-gaap:Revenues"
       contextRef="FY2025" scale="6">416,161</ix:nonFraction></p>
    """

    text = clean_html(html)

    assert "416,161" in text
    assert "us-gaap:Revenues" not in text  # attributes must not leak into text


def test_xbrl_metadata_containers_are_removed_with_contents():
    """`ix:header` / `ix:hidden` hold taxonomy URLs, not readable content."""
    html = """
    <body>
      <ix:header><ix:hidden>http://fasb.org/us-gaap/2025 P1Y 0000320193</ix:hidden></ix:header>
      <ix:references>taxonomy pointers</ix:references>
      <ix:resources>context definitions</ix:resources>
      <p>Visible prose.</p>
    </body>
    """

    text = clean_html(html)

    assert "Visible prose." in text
    assert "fasb.org" not in text
    assert "taxonomy pointers" not in text
    assert "context definitions" not in text


def test_table_row_keeps_label_next_to_its_figures():
    html = """
    <table>
      <tr><td>Europe</td><td>111,032</td><td>10</td><td>%</td></tr>
      <tr><td>Total net sales</td><td>$</td><td>416,161</td><td>6</td><td>%</td></tr>
    </table>
    """

    text = clean_html(html)
    lines = text.split("\n")

    assert "Europe | 111,032 | 10%" in lines
    assert "Total net sales | $416,161 | 6%" in lines


def test_currency_and_percent_cells_are_reattached():
    html = (
        "<table><tr><td>Greater China</td><td>64,377</td>"
        "<td>(4</td><td>)</td><td>%</td></tr></table>"
    )

    assert "Greater China | 64,377 | (4)%" in clean_html(html)


def test_empty_spacer_cells_are_dropped():
    """Filings use empty cells for visual alignment; they add tokens, not meaning."""
    html = "<table><tr><td></td><td>Services</td><td>  </td><td>82,314</td></tr></table>"

    assert "Services | 82,314" in clean_html(html)


def test_table_row_with_no_content_is_skipped():
    html = "<table><tr><td></td><td>  </td></tr><tr><td>Real</td><td>1</td></tr></table>"

    lines = [line for line in clean_html(html).split("\n") if line]

    assert lines == ["Real | 1"]


def test_line_structure_is_preserved_for_the_splitter():
    """Collapsing newlines would leave the recursive splitter nothing to split on."""
    html = "<p>First block.</p><p>Second block.</p><table><tr><td>a</td><td>b</td></tr></table>"

    text = clean_html(html)

    assert "First block." in text
    assert "Second block." in text
    assert "a | b" in text
    assert text.count("\n") >= 2


def test_intra_line_whitespace_is_collapsed_but_newlines_survive():
    html = "<p>spaced     out     prose</p><p>second line</p>"

    text = clean_html(html)

    assert "spaced out prose" in text
    assert "  " not in text
    assert "\n" in text


def test_no_leading_or_trailing_blank_lines():
    text = clean_html("<body>\n\n  <p>Content.</p>\n\n</body>")

    assert text == "Content."


def test_content_free_html_raises():
    with pytest.raises(IngestionError, match="no text"):
        clean_html("<html><head><title>t</title></head><body><script>x=1</script></body></html>")
