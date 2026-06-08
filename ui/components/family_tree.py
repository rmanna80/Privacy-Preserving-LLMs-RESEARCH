"""
Family Tree — editorial SVG visualization.

Renders a static SVG family tree from the People and Relationships
already in the family graph. Generation-stratified, spouse-connected,
parent-child trees descending. Inspired by the editorial wedding-style
mockup (Anderson family) but rendered programmatically.

Algorithm:
  1. Build adjacency from Relationships: spouses, parent_of edges.
  2. Compute generations by BFS from "root couples" (people with no
     known parents in the family). Each level becomes a row.
  3. Within a generation, group spouses adjacently and lay children
     beneath their parents.
  4. Render as SVG: rectangle/oval nodes, lines for parent-child,
     hearts for spouse links, generation labels on the left.

Limitations (documented for honesty):
  - Layout is a best-effort tree, not graph-optimal. Crossing edges
    happen with complex families.
  - No drag/zoom (intentional — this is for screenshots and reports).
  - Maxes out around 30–40 people before the SVG gets cramped.
  - "Generations" are derived from data we have; complex blended
    families may need manual generation override later.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import streamlit as st

from db.models import Person, Relationship
from db.repositories import list_people_in_family, list_relationships_in_family


# ─────────────────────────────────────────────────────────────────────
# Visual constants
# ─────────────────────────────────────────────────────────────────────

NODE_W = 140
NODE_H = 64
NODE_GAP_X = 30
NODE_GAP_Y = 90
GEN_LABEL_W = 110
PADDING = 40

# Brand-aligned palette (cream surface, navy text, gold accents)
COL_BG = "#F8F4EC"
COL_NODE_FILL = "#FFFFFF"
COL_NODE_STROKE_LIVING = "#0B1E3F"
COL_NODE_STROKE_DECEASED = "#9A9A9A"
COL_TEXT = "#0B1E3F"
COL_TEXT_SECONDARY = "#5F7494"
COL_LINE = "#A8884D"
COL_SPOUSE_LINE = "#C9A961"
COL_GEN_LABEL = "#A8884D"


# ─────────────────────────────────────────────────────────────────────
# Layout primitives
# ─────────────────────────────────────────────────────────────────────

@dataclass
class LaidOutNode:
    person: Person
    generation: int
    x: float
    y: float


def _build_graph(
    people: list[Person],
    relationships: list[Relationship],
) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, set[int]]]:
    """Build three indexes from the relationships:
        spouses[pid]   = set of pids married to pid
        children[pid]  = set of pids who are A's children (parent_of edges)
        parents[pid]   = set of pids who are A's parents
    """
    spouses: dict[int, set[int]] = defaultdict(set)
    children: dict[int, set[int]] = defaultdict(set)
    parents: dict[int, set[int]] = defaultdict(set)

    for rel in relationships:
        a, b = rel.person_a_id, rel.person_b_id
        if rel.relationship_type in ("spouse", "ex_spouse"):
            spouses[a].add(b)
            spouses[b].add(a)
        elif rel.relationship_type == "parent_of":
            children[a].add(b)
            parents[b].add(a)

    return spouses, children, parents


def _assign_generations(
    people: list[Person],
    parents: dict[int, set[int]],
    children: dict[int, set[int]],
    spouses: dict[int, set[int]],
) -> dict[int, int]:
    """Assign each person to a generation row.

    Two correctness invariants:
      1. A child's generation = max(parent generations) + 1.
      2. Spouses share the same generation.

    The earlier BFS approach failed invariant (2): a person could be
    pulled down to a lower generation by their parent, while their
    spouse stayed at the original (higher) generation.

    Approach: iterate to a fixed point. Each pass, push children DOWN
    relative to their parents, and pull spouses TOGETHER to the lower
    (numerically higher) of their two generations. Repeat until no
    generation changes — at most ~people-count iterations for a
    well-formed family.
    """
    person_ids = {p.id for p in people}

    # Initial generations: 0 for everyone. The fixed-point iteration
    # below will push people down as needed.
    generation: dict[int, int] = {p.id: 0 for p in people}

    # Cap iterations to avoid pathological cycles (which shouldn't
    # exist in real family data — you can't be your own ancestor —
    # but defensive code never hurt anyone).
    max_iters = len(people) + 5

    for _ in range(max_iters):
        changed = False

        # Pass 1 — push children below their parents
        for parent_id in list(parents.keys()) + [p.id for p in people]:
            # parent_id here is anyone; we look at their children
            for child_id in children.get(parent_id, set()):
                if child_id not in person_ids or parent_id not in person_ids:
                    continue
                required = generation[parent_id] + 1
                if generation[child_id] < required:
                    generation[child_id] = required
                    changed = True

        # Pass 2 — align spouses to the deeper of the two
        for pid, spouse_ids in spouses.items():
            if pid not in person_ids:
                continue
            for sid in spouse_ids:
                if sid not in person_ids:
                    continue
                deeper = max(generation[pid], generation[sid])
                if generation[pid] != deeper:
                    generation[pid] = deeper
                    changed = True
                if generation[sid] != deeper:
                    generation[sid] = deeper
                    changed = True

        if not changed:
            break

    # Normalize: shift everyone up so the highest generation in use is 0.
    # This makes "Generation 1" always the top row, regardless of how
    # the BFS proceeded.
    if generation:
        min_gen = min(generation.values())
        if min_gen != 0:
            generation = {pid: g - min_gen for pid, g in generation.items()}

    return generation


def _order_within_generation(
    people: list[Person],
    generation: dict[int, int],
    spouses: dict[int, set[int]],
    parents: dict[int, set[int]],
) -> dict[int, list[int]]:
    """For each generation, return an ordered list of person_ids.

    Heuristic: group spouses adjacently. Within that, sort by last
    name then first name for stability. This isn't graph-optimal but
    avoids the worst layout edge-crossings on common family shapes.
    """
    by_gen: dict[int, list[Person]] = defaultdict(list)
    for p in people:
        by_gen[generation[p.id]].append(p)

    ordered: dict[int, list[int]] = {}
    for gen, gen_people in by_gen.items():
        # Sort by last name, first name for a stable baseline
        gen_people.sort(key=lambda p: (p.last_name or "", p.first_name or ""))

        # Pair up spouses: walk the list, and when we see a person whose
        # spouse is later in the list, pull the spouse next to them.
        seen: set[int] = set()
        result: list[int] = []
        for person in gen_people:
            if person.id in seen:
                continue
            result.append(person.id)
            seen.add(person.id)
            # Place spouses right after
            for spouse_id in spouses.get(person.id, ()):
                if spouse_id in seen:
                    continue
                # Only if spouse is in same generation
                if generation.get(spouse_id) == gen:
                    result.append(spouse_id)
                    seen.add(spouse_id)

        ordered[gen] = result

    return ordered


def _compute_positions(
    ordered_by_gen: dict[int, list[int]],
    people_by_id: dict[int, Person],
    generation: dict[int, int],
) -> tuple[list[LaidOutNode], int, int]:
    """Assign (x, y) coordinates to every person. Returns (nodes, width, height)."""
    nodes: list[LaidOutNode] = []

    max_per_gen = max((len(ids) for ids in ordered_by_gen.values()), default=1)
    row_width = max_per_gen * (NODE_W + NODE_GAP_X) - NODE_GAP_X

    sorted_gens = sorted(ordered_by_gen.keys())

    for gen in sorted_gens:
        ids = ordered_by_gen[gen]
        row_actual_width = len(ids) * (NODE_W + NODE_GAP_X) - NODE_GAP_X
        # Center this row in the canvas
        start_x = PADDING + GEN_LABEL_W + (row_width - row_actual_width) / 2
        y = PADDING + gen * (NODE_H + NODE_GAP_Y)

        for i, pid in enumerate(ids):
            x = start_x + i * (NODE_W + NODE_GAP_X)
            nodes.append(LaidOutNode(
                person=people_by_id[pid],
                generation=gen,
                x=x,
                y=y,
            ))

    total_width = PADDING * 2 + GEN_LABEL_W + row_width
    total_height = (
        PADDING * 2 + (len(sorted_gens)) * NODE_H
        + max(0, len(sorted_gens) - 1) * NODE_GAP_Y
    )
    return nodes, total_width, total_height


# ─────────────────────────────────────────────────────────────────────
# SVG rendering
# ─────────────────────────────────────────────────────────────────────

def _render_node_svg(node: LaidOutNode) -> str:
    """SVG fragment for one person card."""
    p = node.person
    stroke = (
        COL_NODE_STROKE_DECEASED if p.is_deceased else COL_NODE_STROKE_LIVING
    )
    stroke_dash = ' stroke-dasharray="4 3"' if p.is_deceased else ""

    # Build name lines. Approach:
    #   - If name fits one line (≤18 chars), single centered line.
    #   - If 19–32 chars, split into two lines at the last space before
    #     the midpoint, falls back to mid-string split.
    #   - If 33+ chars, force two lines and shrink the font slightly.
    name = p.display_name
    name_lines, font_size = _break_name(name)

    # DOB / age line
    sub_line = ""
    if p.dob:
        year = p.dob.year
        if p.is_deceased and p.date_of_death:
            sub_line = f"{year}–{p.date_of_death.year}"
        else:
            sub_line = f"b. {year}"
    elif p.is_deceased:
        sub_line = "deceased"

    cx = node.x + NODE_W / 2

    # Vertically center name lines. Card has NODE_H height; lay out
    # name block first, then sub_line a fixed distance below.
    if len(name_lines) == 1:
        name_svg = (
            f'<text x="{cx}" y="{node.y + 26}" text-anchor="middle" '
            f'font-family="Playfair Display, Georgia, serif" '
            f'font-size="{font_size}" font-weight="600" fill="{COL_TEXT}">'
            f'{_escape_xml(name_lines[0])}</text>'
        )
        sub_y = node.y + 46
    else:
        # Two lines: first at y+20, second at y+36
        name_svg = (
            f'<text x="{cx}" y="{node.y + 20}" text-anchor="middle" '
            f'font-family="Playfair Display, Georgia, serif" '
            f'font-size="{font_size}" font-weight="600" fill="{COL_TEXT}">'
            f'{_escape_xml(name_lines[0])}</text>'
            f'<text x="{cx}" y="{node.y + 36}" text-anchor="middle" '
            f'font-family="Playfair Display, Georgia, serif" '
            f'font-size="{font_size}" font-weight="600" fill="{COL_TEXT}">'
            f'{_escape_xml(name_lines[1])}</text>'
        )
        sub_y = node.y + 54

    return f"""
    <g>
      <rect x="{node.x}" y="{node.y}" width="{NODE_W}" height="{NODE_H}"
            rx="6" ry="6"
            fill="{COL_NODE_FILL}" stroke="{stroke}" stroke-width="1.5"
            {stroke_dash}/>
      {name_svg}
      <text x="{cx}" y="{sub_y}" text-anchor="middle"
            font-family="Inter, sans-serif"
            font-size="11" fill="{COL_TEXT_SECONDARY}">
        {_escape_xml(sub_line)}
      </text>
    </g>
    """


def _break_name(name: str) -> tuple[list[str], int]:
    """Return (lines, font_size) for a person's display name.

    Single line for short names. Two lines for longer ones, with font
    shrunk for very long names so they still fit the card width.
    """
    if len(name) <= 18:
        return [name], 13

    # Try to split on a space near the midpoint
    midpoint = len(name) // 2
    # Look for the space closest to the midpoint
    best_split = None
    for i, ch in enumerate(name):
        if ch == " ":
            if best_split is None or abs(i - midpoint) < abs(best_split - midpoint):
                best_split = i

    if best_split is None:
        # No space — hard truncate (very rare for human names)
        line1 = name[:midpoint]
        line2 = name[midpoint:]
    else:
        line1 = name[:best_split]
        line2 = name[best_split + 1:]

    # Shrink font if either line is still too long for the card
    longest = max(len(line1), len(line2))
    if longest <= 14:
        font_size = 13
    elif longest <= 18:
        font_size = 12
    else:
        font_size = 11

    return [line1, line2], font_size


def _render_spouse_line(a: LaidOutNode, b: LaidOutNode) -> str:
    """Horizontal line + small heart between two spouses."""
    # Make sure a is to the left of b
    if a.x > b.x:
        a, b = b, a

    y = a.y + NODE_H / 2
    x1 = a.x + NODE_W
    x2 = b.x
    mid_x = (x1 + x2) / 2

    return f"""
    <line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}"
          stroke="{COL_SPOUSE_LINE}" stroke-width="1.5"/>
    <text x="{mid_x}" y="{y + 4}" text-anchor="middle"
          font-size="12" fill="{COL_SPOUSE_LINE}">♥</text>
    """


def _render_parent_child_line(parent: LaidOutNode, child: LaidOutNode) -> str:
    """T-shaped line: down from parent, across, down to child."""
    px = parent.x + NODE_W / 2
    py = parent.y + NODE_H
    cx = child.x + NODE_W / 2
    cy = child.y
    mid_y = (py + cy) / 2

    return f"""
    <path d="M {px} {py} L {px} {mid_y} L {cx} {mid_y} L {cx} {cy}"
          stroke="{COL_LINE}" stroke-width="1.2" fill="none"/>
    """


def _render_generation_label(gen: int, y: float) -> str:
    cy = y + NODE_H / 2
    return f"""
    <text x="{PADDING}" y="{cy + 4}" text-anchor="start"
          font-family="Inter, sans-serif"
          font-size="11" font-weight="600" fill="{COL_GEN_LABEL}"
          letter-spacing="2">
      GENERATION {gen + 1}
    </text>
    """


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


def _build_svg(
    nodes: list[LaidOutNode],
    spouses: dict[int, set[int]],
    children: dict[int, set[int]],
    width: int,
    height: int,
) -> str:
    nodes_by_id = {n.person.id: n for n in nodes}
    generations_seen: set[int] = set()

    parts: list[str] = []

    # Background
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" '
        f'fill="{COL_BG}" rx="12" ry="12"/>'
    )

    # Generation labels
    for n in nodes:
        if n.generation not in generations_seen:
            generations_seen.add(n.generation)
            parts.append(_render_generation_label(n.generation, n.y))

    # Parent-child lines (drawn behind nodes)
    drawn_parent_edges: set[tuple[int, int]] = set()
    for parent_id, child_ids in children.items():
        if parent_id not in nodes_by_id:
            continue
        for child_id in child_ids:
            if child_id not in nodes_by_id:
                continue
            edge = (parent_id, child_id)
            if edge in drawn_parent_edges:
                continue
            drawn_parent_edges.add(edge)
            parts.append(_render_parent_child_line(
                nodes_by_id[parent_id],
                nodes_by_id[child_id],
            ))

    # Spouse lines
    drawn_spouse_pairs: set[frozenset[int]] = set()
    for pid, spouse_ids in spouses.items():
        if pid not in nodes_by_id:
            continue
        for sid in spouse_ids:
            if sid not in nodes_by_id:
                continue
            pair = frozenset({pid, sid})
            if pair in drawn_spouse_pairs:
                continue
            drawn_spouse_pairs.add(pair)
            # Only draw if same generation (else it's not a horizontal line)
            if nodes_by_id[pid].generation == nodes_by_id[sid].generation:
                parts.append(_render_spouse_line(
                    nodes_by_id[pid], nodes_by_id[sid]
                ))

    # Nodes (drawn last so they sit on top of lines)
    for n in nodes:
        parts.append(_render_node_svg(n))

    body = "\n".join(parts)
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         viewBox="0 0 {width} {height}"
         preserveAspectRatio="xMidYMid meet"
         width="100%"
         height="100%"
         style="background: {COL_BG}; border-radius: 12px; display: block;">
      {body}
    </svg>
    """


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def render_family_tree(family_id: int) -> None:
    """Render the family tree for a family."""
    people = list_people_in_family(family_id)
    relationships = list_relationships_in_family(family_id)

    st.markdown("### Family Tree")
    st.caption(
        "An editorial view of this family's structure, generated from "
        "the People and Relationships in Key People & Orgs."
    )

    if not people:
        st.info(
            "Add people in **Key People & Orgs → People** to see them here."
        )
        return

    if len(people) == 1:
        # Single-person fallback — render just the node
        st.info(
            "A family tree needs at least two people connected by a "
            "relationship. Add another person and a relationship "
            "(spouse, parent_of) in Key People & Orgs."
        )
        return

    # Build the layout
    spouses, children, parents = _build_graph(people, relationships)
    generation = _assign_generations(people, parents, children, spouses)
    ordered = _order_within_generation(people, generation, spouses, parents)
    people_by_id = {p.id: p for p in people}
    nodes, width, height = _compute_positions(ordered, people_by_id, generation)

    svg = _build_svg(nodes, spouses, children, width, height)

    # # Render the SVG. Streamlit's markdown handles raw SVG well.
    # st.markdown(svg, unsafe_allow_html=True)

        # Render the SVG inside an iframe so Streamlit's HTML sanitizer
    # doesn't strip it. st.markdown() will silently drop SVG content
    # even with unsafe_allow_html=True; components.v1.html doesn't.
    import streamlit.components.v1 as components

    # Compute a fixed iframe height that's tall enough to feel like a
    # poster but never causes scrolling. The SVG scales to fit thanks
    # to its viewBox + preserveAspectRatio.
    aspect_ratio = height / width  # the tree's natural shape
    # Cap iframe height at 650px so the page itself stays scannable,
    # but allow tall narrow trees to extend down to 800px.
    iframe_height = min(800, max(400, int(700 * aspect_ratio + 60)))

    components.html(
        f"""
        <div style="background:{COL_BG}; border-radius:12px; padding:8px;
                    height:{iframe_height - 16}px; box-sizing:border-box;
                    display:flex; align-items:center; justify-content:center;">
          {svg}
        </div>
        """,
        height=iframe_height,
        scrolling=False,
    )

    # Caption + helpful next-actions
    st.markdown("---")
    col_legend, col_actions = st.columns([3, 2])

    with col_legend:
        st.markdown("**Legend**")
        st.markdown(
            f"- **Solid border** — living family member  \n"
            f"- **Dashed border** — deceased  \n"
            f"- **♥ horizontal line** — spouse  \n"
            f"- **Vertical line** — parent → child"
        )

    with col_actions:
        st.markdown("**Tips**")
        st.caption(
            "If the tree looks crowded or has crossing lines, the "
            "underlying People & Relationships data may need cleanup. "
            "Spouses must be marked as 'spouse'; parent-child must be "
            "'parent_of' (direction matters)."
        )

    # Download as SVG
    st.download_button(
        label="⬇  Download as SVG",
        data=svg,
        file_name=f"family_tree_{family_id}.svg",
        mime="image/svg+xml",
        use_container_width=False,
    )
