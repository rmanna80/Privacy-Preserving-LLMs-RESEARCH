"""
Angel — visual identity for the platform.

Applied via a single inject_theme() call early in the page render. Sets
typography, color palette, button styling, sidebar appearance, and a few
component-specific overrides.

Brand:
  Name:       Angel
  Tagline:    Database for Family Wealth
  Palette:    Deep navy (#0B1E3F) + champagne gold (#C9A961)
              Cream surface (#F8F4EC) for editorial pages (family tree)
  Type:       Playfair Display (serif) for headings
              Inter (sans) for body

Aesthetic target: "professionally clean", aiming at private-bank /
single-family-office tone rather than fintech bright. The editorial
look in the family tree mockup is the destination, this is the
foundation to grow toward it.
"""

from __future__ import annotations

import streamlit as st


# ─────────────────────────────────────────────────────────────────────
# Color tokens — referenced by name elsewhere if needed
# ─────────────────────────────────────────────────────────────────────

class Color:
    NAVY_900 = "#0B1E3F"
    NAVY_800 = "#15294D"
    NAVY_700 = "#22385F"
    NAVY_400 = "#5F7494"

    GOLD_600 = "#A8884D"
    GOLD_500 = "#C9A961"   # primary accent
    GOLD_400 = "#D8BC7E"

    CREAM = "#F8F4EC"
    SURFACE = "#0F2347"
    SURFACE_ALT = "#15294D"
    TEXT_ON_DARK = "#E9ECF4"
    TEXT_MUTED_ON_DARK = "#9AA8C0"

    DANGER = "#C66666"
    SUCCESS = "#7FB69A"


# ─────────────────────────────────────────────────────────────────────
# Brand strings
# ─────────────────────────────────────────────────────────────────────

BRAND_NAME = "Angel"
BRAND_TAGLINE = "Database for Family Wealth"
BRAND_GLYPH = "👼"  # placeholder until the actual logo is wired in


# ─────────────────────────────────────────────────────────────────────
# CSS — injected once per page
# ─────────────────────────────────────────────────────────────────────

_CSS = f"""
<style>
/* ───── Web fonts ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

/* ───── Global background and text ──────────────────────────────── */
.stApp {{
    background: linear-gradient(180deg, {Color.NAVY_900} 0%, {Color.NAVY_800} 100%);
    color: {Color.TEXT_ON_DARK};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

/* Main content area */
section.main > div.block-container {{
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1400px;
}}

/* ───── Headings — editorial serif ──────────────────────────────── */
h1, h2, h3 {{
    font-family: 'Playfair Display', Georgia, serif !important;
    color: {Color.TEXT_ON_DARK} !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}}

h1 {{ font-size: 2.4rem !important; }}
h2 {{ font-size: 1.7rem !important; }}
h3 {{ font-size: 1.3rem !important; }}

/* Caption (st.caption) */
.stApp [data-testid="stCaptionContainer"],
.stApp .stCaption {{
    color: {Color.TEXT_MUTED_ON_DARK} !important;
    font-size: 0.9rem;
}}

/* ───── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: {Color.NAVY_900};
    border-right: 1px solid rgba(201,169,97,0.12);
}}

section[data-testid="stSidebar"] * {{
    color: {Color.TEXT_ON_DARK};
}}

/* Sidebar buttons styled as nav links */
section[data-testid="stSidebar"] div.stButton > button {{
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    background: transparent;
    border: none;
    padding: 10px 14px;
    margin: 1px 0;
    border-radius: 8px;
    font-size: 0.92rem;
    font-weight: 400;
    color: {Color.TEXT_ON_DARK} !important;
    font-family: 'Inter', sans-serif !important;
    transition: background 0.15s ease;
}}

section[data-testid="stSidebar"] div.stButton > button:hover {{
    background: rgba(201,169,97,0.08);
    color: {Color.GOLD_400} !important;
}}

section[data-testid="stSidebar"] div.nav-active div.stButton > button {{
    background: rgba(201,169,97,0.15);
    color: {Color.GOLD_500} !important;
    font-weight: 600;
    border-left: 3px solid {Color.GOLD_500};
    padding-left: 11px;
}}

/* Sidebar section captions */
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    color: {Color.GOLD_500} !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 14px 0 4px 14px;
}}

/* ───── Primary buttons (main content) ──────────────────────────── */
.stApp section.main div.stButton > button {{
    background: {Color.GOLD_500};
    color: {Color.NAVY_900};
    border: none;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    border-radius: 6px;
    padding: 0.5rem 1.2rem;
    transition: all 0.15s ease;
}}

.stApp section.main div.stButton > button:hover {{
    background: {Color.GOLD_400};
    color: {Color.NAVY_900};
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(201,169,97,0.25);
}}

.stApp section.main div.stButton > button:focus,
.stApp section.main div.stButton > button:focus:not(:active) {{
    border-color: {Color.GOLD_500};
    color: {Color.NAVY_900};
}}

/* Secondary button (non-primary) — outlined */
.stApp section.main div.stButton > button[kind="secondary"] {{
    background: transparent;
    color: {Color.GOLD_500};
    border: 1px solid {Color.GOLD_500};
}}

/* ───── Inputs ──────────────────────────────────────────────────── */
.stApp input[type="text"],
.stApp input[type="password"],
.stApp input[type="number"],
.stApp textarea,
.stApp .stSelectbox > div > div,
.stApp .stMultiSelect > div > div,
.stApp .stDateInput input,
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea {{
    background: {Color.SURFACE} !important;
    color: {Color.TEXT_ON_DARK} !important;
    border: 1px solid rgba(201,169,97,0.18) !important;
    border-radius: 6px !important;
}}

.stApp input[type="text"]:focus,
.stApp input[type="password"]:focus,
.stApp textarea:focus {{
    border-color: {Color.GOLD_500} !important;
    box-shadow: 0 0 0 2px rgba(201,169,97,0.15) !important;
}}

/* Input labels */
.stApp label, .stApp [data-testid="stWidgetLabel"] {{
    color: {Color.TEXT_MUTED_ON_DARK} !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}}

/* ───── Containers / cards ──────────────────────────────────────── */
.stApp [data-testid="stVerticalBlockBorderWrapper"] {{
    background: {Color.SURFACE_ALT};
    border: 1px solid rgba(201,169,97,0.12) !important;
    border-radius: 10px;
    padding: 1rem;
}}

/* st.metric */
.stApp [data-testid="stMetric"] {{
    background: {Color.SURFACE_ALT};
    border: 1px solid rgba(201,169,97,0.10);
    border-radius: 10px;
    padding: 1rem 1.2rem;
}}

.stApp [data-testid="stMetricLabel"] {{
    color: {Color.TEXT_MUTED_ON_DARK} !important;
    font-size: 0.8rem !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

.stApp [data-testid="stMetricValue"] {{
    color: {Color.TEXT_ON_DARK} !important;
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 2rem !important;
    font-weight: 600;
}}

/* Info / warning / error / success alerts */
.stApp [data-testid="stAlert"] {{
    background: {Color.SURFACE_ALT};
    border-left: 3px solid {Color.GOLD_500};
    color: {Color.TEXT_ON_DARK} !important;
    border-radius: 6px;
}}

/* ───── Tabs ────────────────────────────────────────────────────── */
.stApp [data-baseweb="tab-list"] {{
    border-bottom: 1px solid rgba(201,169,97,0.18);
}}

.stApp [data-baseweb="tab"] {{
    color: {Color.TEXT_MUTED_ON_DARK} !important;
    font-family: 'Inter', sans-serif !important;
}}

.stApp [data-baseweb="tab"][aria-selected="true"] {{
    color: {Color.GOLD_500} !important;
}}

.stApp [data-baseweb="tab-highlight"] {{
    background: {Color.GOLD_500} !important;
}}

/* ───── Tables & dataframes ─────────────────────────────────────── */
.stApp [data-testid="stDataFrame"] {{
    background: {Color.SURFACE_ALT};
    border-radius: 8px;
}}

/* ───── Code blocks (used for masked SSNs etc.) ─────────────────── */
.stApp code,
.stApp [data-testid="stCodeBlock"] {{
    background: {Color.SURFACE} !important;
    color: {Color.GOLD_400} !important;
    border: 1px solid rgba(201,169,97,0.15);
    border-radius: 5px;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}}

/* ───── Brand header ────────────────────────────────────────────── */
.angel-brand-header {{
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(201,169,97,0.15);
}}

.angel-brand-name {{
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: {Color.GOLD_500};
    letter-spacing: -0.01em;
}}

.angel-brand-tagline {{
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: {Color.TEXT_MUTED_ON_DARK};
    text-transform: uppercase;
    letter-spacing: 0.12em;
}}

/* ───── Editorial page surface (Family Tree etc.) ───────────────── */
.editorial-page {{
    background: {Color.CREAM};
    color: {Color.NAVY_900};
    padding: 3rem 2rem;
    border-radius: 12px;
    margin-top: 1rem;
}}

.editorial-page h1,
.editorial-page h2,
.editorial-page h3 {{
    color: {Color.NAVY_900} !important;
}}

/* ───── Misc cleanup ────────────────────────────────────────────── */

/* Hide the streamlit hamburger / "Deploy" hover in top right when
   you don't want it for screenshots. Leave commented in dev. */
/* #MainMenu {{ visibility: hidden; }} */
/* header {{ visibility: hidden; }} */

</style>
"""


def inject_theme() -> None:
    """Call once near the top of each page render."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_brand_header() -> None:
    """Render the Angel brand strip — uses static/angel_logo.png if present,
    falls back to the emoji glyph otherwise. Drop a logo file at
    static/angel_logo.png to swap to the real logo with no code change."""
    from pathlib import Path
    import base64
 
    logo_path = Path("static/angel_logo.png")
    if logo_path.exists():
        try:
            encoded = base64.b64encode(logo_path.read_bytes()).decode()
            logo_html = (
                f'<img src="data:image/png;base64,{encoded}" '
                f'style="height:32px; vertical-align:middle; '
                f'margin-right:8px;" alt="Angel">'
            )
        except Exception:
            logo_html = BRAND_GLYPH
    else:
        logo_html = BRAND_GLYPH
 
    st.markdown(
        f"""
        <div class="angel-brand-header">
          <div>
            <div class="angel-brand-name">{logo_html} {BRAND_NAME}</div>
            <div class="angel-brand-tagline">{BRAND_TAGLINE}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )