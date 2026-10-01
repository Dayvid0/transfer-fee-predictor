"""Visual helper tests: the artwork is generated locally, is well-formed, and
draws every colour from the one shared palette rather than hardcoding hexes."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import plotly.graph_objects as go
import pytest

from src import config as C
from src import ui
from src import visuals as V

HEX = re.compile(r"#[0-9A-Fa-f]{6}")


# --- purity and well-formedness ---------------------------------------------


def test_svg_builders_produce_well_formed_xml():
    builders = [
        V.pitch_svg(),
        V.pitch_svg("Custom title"),
        V.position_pitch_svg("ST"),
        V.position_pitch_svg("GK"),
        V.avatar_svg("E. Haaland", "ST"),
        V.avatar_svg("Cher", "GK", 48),
    ]
    for markup in builders:
        root = ET.fromstring(markup)
        assert root.tag.endswith("svg"), markup[:60]
        # Self-contained: no external href, so nothing is ever fetched.
        assert "http://" not in markup.replace(
            'xmlns="http://www.w3.org/2000/svg"', ""
        )
        assert "https://" not in markup


@pytest.mark.parametrize("position", C.POSITIONS)
def test_every_position_draws_a_valid_pitch(position: str):
    markup = V.position_pitch_svg(position)
    assert ET.fromstring(markup).tag.endswith("svg")
    assert f">{position}<" in markup, "position label should be visible"
    assert V.position_colour(position) == ui.PALETTE["position"][position]


def test_visual_builders_are_deterministic():
    """No randomness: the same arguments must give byte-identical output."""
    assert V.pitch_svg() == V.pitch_svg()
    assert V.position_pitch_svg("CM") == V.position_pitch_svg("CM")
    assert V.avatar_svg("E. Haaland", "ST") == V.avatar_svg("E. Haaland", "ST")
    assert V.player_card("A", "ST", "La Liga", 25, "EUR 1") == V.player_card(
        "A", "ST", "La Liga", 25, "EUR 1"
    )


# --- palette discipline -----------------------------------------------------


def test_every_colour_comes_from_the_shared_palette():
    """No view or SVG may introduce a colour the palette does not define."""
    allowed = {c.upper() for c in HEX.findall(str(ui.PALETTE["position"]))}
    for key, value in ui.PALETTE.items():
        if key == "position":
            continue
        allowed.add(value.upper())
    allowed |= {c.upper() for c in ui.SERIES}

    for markup in (
        V.pitch_svg(),
        V.position_pitch_svg("ST"),
        V.avatar_svg("E. Haaland", "ST"),
        V.player_card("E. Haaland", "ST", "Premier League", 26, "EUR 1"),
    ):
        for colour in HEX.findall(markup):
            assert colour.upper() in allowed, f"{colour} is not in PALETTE"


def test_css_defines_every_class_the_svg_helpers_emit():
    """A card or hero that ships without its CSS renders unstyled."""
    for cls in ("hero", "pcard", "pcard-row", "pcard-top", "pcard-name",
                "pcard-sub", "pcard-fee"):
        assert f".{cls}" in ui.CSS, f".{cls} is emitted but never styled"
    assert ui.CSS.count("<style>") == 1
    assert len(ui.CSS.splitlines()) <= 60, "CSS should stay compact"


def test_series_colours_start_with_the_accent():
    assert ui.SERIES[0] == ui.PALETTE["accent"]
    assert len(ui.SERIES) >= 6, "categorical plots need at least 6 colours"


# --- content ----------------------------------------------------------------


def test_initials_handle_real_name_shapes():
    assert V.initials("E. Haaland") == "EH"
    assert V.initials("Kylian Mbappe") == "KM"
    assert V.initials("Cher") == "CH"
    assert V.initials("") == "?"
    assert V.initials("Xherdan Shaqiri") == "XS"


def test_league_flag_is_an_emoji_not_an_image_url():
    for league in C.LEAGUES:
        flag = V.league_flag(league)
        assert flag and "http" not in flag, league
        assert flag.encode("utf-8").decode("utf-8") == flag
    # Unknown leagues still get a neutral marker rather than raising.
    assert V.league_flag("Nowhere League") == "\U0001F3E0"


def test_player_card_shows_the_fee_the_caller_passed():
    """The card formats a label; it must not recompute or re-price the fee."""
    markup = V.player_card(
        "Player 42", "ST", "Premier League", 26, "EUR 12,345,678"
    )
    assert markup.count("Player 42") == 1
    assert "EUR 12,345,678" in markup
    assert "Premium" not in markup


def test_top_cards_renders_one_card_per_row():
    rows = [
        {"name": f"Player {i}", "position": "ST", "league": "La Liga",
         "age": 24 + i, "fee_eur": 1_000_000 * (i + 1)}
        for i in range(3)
    ]
    markup = V.top_cards(rows)
    assert markup.count('class="pcard"') == 3
    assert "EUR 1,000,000" in markup and "EUR 3,000,000" in markup
    assert V.top_cards([]).count('class="pcard"') == 0


# --- the fee gauge ----------------------------------------------------------


def test_fee_gauge_is_a_plotly_indicator_with_the_value_placed():
    fig = V.fee_gauge(12_000_000, 5_000_000)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "indicator"
    assert fig.data[0].value == 12_000_000
    # The threshold marks the median, so the needle reads against it.
    assert fig.data[0].gauge.threshold.value == 5_000_000


def test_fee_gauge_axis_covers_the_value_it_reports():
    for value, median in ((12_000_000, 5_000_000), (500_000, 5_000_000), (0.0, 5_000_000)):
        fig = V.fee_gauge(value, median)
        low, high = fig.data[0].gauge.axis.range
        assert low == 0
        assert high > value, f"value {value} fell off a {high} axis"


def test_fee_gauge_survives_a_missing_median():
    """A season with no paid rows must not produce a zero-division or NaN."""
    fig = V.fee_gauge(1_000_000, 0.0)
    low, high = fig.data[0].gauge.axis.range
    assert low == 0 and high > 0
    assert fig.data[0].gauge.threshold.value == 1.0


def test_plot_layout_is_dark_transparent_and_overridable():
    layout = V.plot_layout(height=300, margin=dict(l=1, r=2, t=3, b=4))
    assert layout["template"] == "plotly_dark"
    assert layout["paper_bgcolor"] == "rgba(0,0,0,0)"
    assert layout["plot_bgcolor"] == "rgba(0,0,0,0)"
    assert layout["font"]["color"] == ui.PALETTE["text"]
    assert layout["height"] == 300, "overrides must win"
    assert layout["margin"] == {"l": 1, "r": 2, "t": 3, "b": 4}
    # The default layout must not be mutated between calls.
    assert V.plot_layout()["margin"] != layout["margin"]
