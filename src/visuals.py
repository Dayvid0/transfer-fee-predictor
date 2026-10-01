"""Locally generated artwork: pitch, avatars, cards and the fee gauge.

Every function here is pure: same arguments, same string or figure back, with no
I/O, no network and no randomness. All graphics are inline SVG or Plotly
objects, so nothing is fetched and no external font is needed.
"""
from __future__ import annotations

import plotly.graph_objects as go

from src.ui import PALETTE

XL, XR, MID, TOP = 40, 960, 500, 20

# Where each position sits on a half-pitch, as fractions of the pitch box.
POSITION_XY = {
    "GK": (0.045, 0.50), "CB": (0.17, 0.50), "LB": (0.21, 0.20), "RB": (0.21, 0.80),
    "DM": (0.33, 0.50), "CM": (0.45, 0.50), "AM": (0.60, 0.50),
    "LW": (0.70, 0.20), "RW": (0.70, 0.80), "ST": (0.78, 0.50),
}

# Country flag emoji per league. Inline unicode, never an image URL.
LEAGUE_FLAG = {
    "Premier League": "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F",
    "La Liga": "\U0001F1EA\U0001F1F8", "Serie A": "\U0001F1EE\U0001F1F9",
    "Bundesliga": "\U0001F1E9\U0001F1EA", "Ligue 1": "\U0001F1EB\U0001F1F7",
    "Eredivisie": "\U0001F1F3\U0001F1F1", "Primeira Liga": "\U0001F1F5\U0001F1F9",
}

SVG = 'xmlns="http://www.w3.org/2000/svg" role="img"'


def league_flag(league: str) -> str:
    """Flag emoji for a league, falling back to a neutral ball."""
    return LEAGUE_FLAG.get(league, "\U0001F3E0")


def position_colour(position: str) -> str:
    """One colour per position, drawn from the shared palette."""
    return PALETTE["position"].get(position, PALETTE["muted"])


def plot_layout(**overrides) -> dict:
    """Shared plotly layout so every chart in the app looks the same."""
    base = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": PALETTE["text"]},
        "margin": {"l": 10, "r": 10, "t": 60, "b": 10},
        "hoverlabel": {"bgcolor": "#141D2E"},
    }
    base.update(overrides)
    return base


def _turf(height: int, gid: str) -> str:
    """The turf background every pitch shares."""
    return (
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{PALETTE["pitch_a"]}"/>'
        f'<stop offset="100%" stop-color="{PALETTE["pitch_b"]}"/>'
        f'</linearGradient></defs>'
        f'<rect width="{XR + 40}" height="{height}" fill="url(#{gid})"/>'
    )


def _marks(accent: str, height: int) -> str:
    """Shared line work: outline, halfway line, centre circle, boxes, spots."""
    bot, span = height - TOP, height - 2 * TOP
    l = f'stroke="{accent}" stroke-width="2" fill="none" opacity="0.85"'
    s = f'fill="{accent}"'
    wide, deep = span * 0.588, (XR - XL) * 0.13
    six, spot = span * 0.353, deep * 0.71
    return "".join([
        f'<rect x="{XL}" y="{TOP}" width="{XR - XL}" height="{span}" rx="4" {l}/>',
        f'<line x1="{MID}" y1="{TOP}" x2="{MID}" y2="{bot}" {l}/>',
        f'<circle cx="{MID}" cy="{bot / 2}" r="{span * 0.152:.0f}" {l}/>',
        f'<circle cx="{MID}" cy="{bot / 2:.0f}" r="3" {s}/>',
        f'<rect x="{XL}" y="{bot / 2 - wide / 2:.0f}" width="{deep:.0f}" height="{wide:.0f}" {l}/>',
        f'<rect x="{XL}" y="{bot / 2 - six / 2:.0f}" width="{six * 0.33:.0f}" height="{six:.0f}" {l}/>',
        f'<circle cx="{XL + spot:.0f}" cy="{bot / 2:.0f}" r="2" {s}/>',
        f'<rect x="{XR - deep:.0f}" y="{bot / 2 - wide / 2:.0f}" width="{deep:.0f}" height="{wide:.0f}" {l}/>',
        f'<rect x="{XR - six * 0.33:.0f}" y="{bot / 2 - six / 2:.0f}" width="{six * 0.33:.0f}" height="{six:.0f}" {l}/>',
        f'<circle cx="{XR - spot:.0f}" cy="{bot / 2:.0f}" r="2" {s}/>',
    ])


def pitch_svg(title: str = "Transfer fee predictor", accent: str = PALETTE["accent"]) -> str:
    """Inline SVG of a football pitch carrying the project title."""
    label = (
        f'<text x="{MID}" y="172" text-anchor="middle" fill="{PALETTE["text"]}" '
        f'font-family="sans-serif" font-size="38" font-weight="700">{title}</text>'
        f'<text x="{MID}" y="222" text-anchor="middle" fill="{accent}" '
        f'font-family="sans-serif" font-size="19" letter-spacing="3">SYNTHETIC MARKET</text>'
    )
    return (
        f'<svg viewBox="0 0 {XR + 40} 380" width="100%" {SVG} aria-label="{title}">'
        + _turf(380, "turf") + _marks(accent, 380) + label + "</svg>"
    )


def position_pitch_svg(position: str, accent: str = PALETTE["accent"]) -> str:
    """The pitch with the chosen position marked by a filled dot."""
    frac_x, frac_y = POSITION_XY.get(position, (0.5, 0.5))
    cx = XL + frac_x * (XR - XL)
    cy = TOP + frac_y * (300 - 2 * TOP)
    colour = position_colour(position)
    dot = (
        f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="26" fill="{colour}" opacity="0.28"/>'
        f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="11" fill="{colour}" '
        f'stroke="{PALETTE["text"]}" stroke-width="2"/>'
        f'<text x="{cx:.0f}" y="{cy - 36:.0f}" text-anchor="middle" fill="{PALETTE["text"]}" '
        f'font-family="sans-serif" font-size="17" font-weight="700">{position}</text>'
    )
    return (
        f'<svg viewBox="0 0 {XR + 40} 300" width="100%" {SVG} aria-label="Pitch showing {position}">'
        + _turf(300, "turf2") + _marks(accent, 300) + dot + "</svg>"
    )


def initials(name: str) -> str:
    """Up to two initials from a player's name, e.g. 'E. Haaland' -> 'EH'."""
    parts = [p for p in str(name).replace(".", " ").split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def avatar_svg(name: str, position: str, size: int = 72) -> str:
    """A circular initials badge, coloured by position."""
    return (
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" {SVG} '
        f'aria-label="{initials(name)} {position}">'
        f'<circle cx="{size / 2}" cy="{size / 2}" r="{size / 2}" '
        f'fill="{position_colour(position)}"/>'
        f'<text x="{size / 2}" y="{size / 2}" text-anchor="middle" dominant-baseline="central" '
        f'fill="{PALETTE["on_accent"]}" font-family="sans-serif" font-size="{int(size * 0.34)}" '
        f'font-weight="700">{initials(name)}</text></svg>'
    )


def player_card(name: str, position: str, league: str, age: int, fee_label: str) -> str:
    """One player card: avatar, flag, identity and a large fee badge."""
    return (
        f'<div class="pcard"><div class="pcard-top">{avatar_svg(name, position, 56)}'
        f'<div><div class="pcard-name">{name}</div><div class="pcard-sub">'
        f'{league_flag(league)} {league} &middot; {position} &middot; age {age}</div></div></div>'
        f'<div class="pcard-fee">{fee_label}</div></div>'
    )


def top_cards(cards: list[dict], fee_key: str = "fee_eur") -> str:
    """A row of player cards for a ranked list of predictions."""
    body = "".join(
        player_card(c["name"], c["position"], c["league"], int(c["age"]),
                    f'EUR {float(c[fee_key]):,.0f}') for c in cards
    )
    return f'<div class="pcard-row">{body}</div>'


def fee_gauge(value: float, median: float,
              title: str = "Predicted fee vs season median") -> go.Figure:
    """Plotly gauge placing a predicted fee against the season's median."""
    median = float(median) or 1.0
    ceiling = max(median * 2.5, float(value) * 1.15)
    steps = [
        {"range": [0, 0.5], "color": PALETTE["muted"]},
        {"range": [0.5, 1.0], "color": PALETTE["secondary"]},
        {"range": [1.0, 1.75], "color": PALETTE["accent_soft"]},
        {"range": [1.75, ceiling / median], "color": PALETTE["accent"]},
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=float(value),
        number={"suffix": " EUR", "valueformat": ",.0f", "font": {"size": 26}},
        delta={"reference": median, "relative": True, "valueformat": ".0%"},
        gauge={
            "axis": {"range": [0, ceiling], "tickfont": {"size": 11}},
            "bar": {"color": PALETTE["accent"]},
            "steps": steps,
            "threshold": {"line": {"color": PALETTE["text"], "width": 3},
                          "value": median, "thickness": 0.8},
        },
    ))
    fig.update_layout(title=title, **plot_layout(height=280, margin=dict(l=20, r=20, t=50, b=20)))
    return fig
