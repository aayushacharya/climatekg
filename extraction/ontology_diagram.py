"""
Generate the ClimatePub4KG ontology diagram (node types + relationship types).

The diagram is the schema-level view of the KG that is published on
https://climatepub4kg.github.io (replaces the older `paper_combined_ontology.png`).

The ontology is curated below in ``CLUSTERS`` / ``EDGES`` so the layout can be
controlled, but ``--verify`` cross-checks it against the live Neo4j instance and
against ``consts.ALL_RELATIONSHIP_SOURCE_TO_TARGET`` so it cannot silently drift
from the graph.

Outputs `.dot`, `.svg` and `.png` into output/ontology_diagram/.

Usage (from this directory):
    python ontology_diagram.py                     # regenerate the diagram
    python ontology_diagram.py --verify            # also diff against live Neo4j
    python ontology_diagram.py --png-width 6000    # larger raster
"""

import argparse
import os
import re
import shlex
import shutil
import subprocess
from collections import defaultdict

from consts import ALL_RELATIONSHIP_SOURCE_TO_TARGET

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "ontology_diagram")

# --------------------------------------------------------------------------------------
# Provenance palette. A node type can come from more than one source, in which case the
# fill is striped over each of its sources (in the order listed here).
# --------------------------------------------------------------------------------------
SOURCES = {
    "esgf": ("ESGF / CMIP6 controlled vocabulary", "#8ecae6"),
    "geonames": ("GeoNames gazetteer", "#c9e4a6"),
    "gcmd": ("NASA GCMD keywords", "#a2d2ff"),
    "paper": ("LLM-extracted from papers", "#ffd166"),
    "openalex": ("OpenAlex publication metadata", "#e5c6f2"),
}

# --------------------------------------------------------------------------------------
# Node types, grouped into the clusters drawn in the diagram.
# provenance -> tuple of keys into SOURCES.
# --------------------------------------------------------------------------------------
CLUSTERS = [
    {
        "key": "pub",
        "label": "Publication metadata",
        "color": "#7b2d8b",
        "bg": "#faf3fc",
        "nodes": [
            ("Paper", ("openalex",)),
            ("Author", ("openalex",)),
            ("Venue", ("openalex",)),
            ("Field", ("openalex",)),
            ("Keyword", ("openalex",)),
        ],
    },
    {
        "key": "claim",
        "label": "Research content (extracted from papers)",
        "color": "#b8860b",
        "bg": "#fffbf0",
        "nodes": [
            ("Task", ("paper",)),
            ("Problem", ("paper",)),
            ("Method", ("paper",)),
            ("Result", ("paper",)),
            ("Innovation", ("paper",)),
            ("EvaluationMetric", ("paper",)),
            ("ObservationalDataset", ("paper",)),
        ],
    },
    {
        "key": "esgf",
        "label": "ESGF / CMIP6 simulation metadata",
        "color": "#1f6f8b",
        "bg": "#f2fafd",
        "nodes": [
            ("Source", ("esgf",)),
            ("SourceType", ("esgf",)),
            ("SourceComponent", ("esgf", "paper")),
            ("Variable", ("esgf",)),
            ("Realm", ("esgf",)),
            ("Experiment", ("esgf",)),
            ("SubExperiment", ("esgf",)),
            ("ExperimentFamily", ("esgf",)),
            ("Activity", ("esgf",)),
            ("Institute", ("esgf",)),
            ("Project", ("esgf",)),
            ("MIPEra", ("esgf",)),
            ("Member", ("esgf",)),
            ("Ensemble", ("esgf",)),
            ("Forcing", ("esgf",)),
            ("Frequency", ("esgf",)),
            ("Resolution", ("esgf",)),
            ("Domain", ("esgf",)),
            ("RCM", ("esgf",)),
        ],
    },
    {
        "key": "modelchar",
        "label": "Model characteristics (extracted from papers)",
        "color": "#b8860b",
        "bg": "#fffbf0",
        "nodes": [
            ("PhysicalScheme", ("paper",)),
            ("PhysicalFeature", ("paper",)),
            ("SimulationType", ("paper",)),
            ("Metric", ("paper",)),
        ],
    },
    {
        "key": "phenom",
        "label": "Earth-science phenomena & observation (GCMD)",
        "color": "#2a7f5f",
        "bg": "#f4fbf7",
        "nodes": [
            ("Natural_Hazard", ("gcmd", "paper")),
            ("NaturalHazardType", ("gcmd",)),
            ("Weather_Event", ("gcmd", "paper")),
            ("Teleconnection", ("gcmd",)),
            ("Ocean_Circulation", ("gcmd",)),
            ("Instrument", ("gcmd", "paper")),
            ("Platform", ("gcmd", "paper")),
        ],
    },
    {
        "key": "geo",
        "label": "Geography",
        "color": "#4c7a1f",
        "bg": "#f7fbf2",
        "nodes": [
            ("Continent", ("geonames",)),
            ("Country", ("geonames",)),
            ("Country_Subdivision", ("geonames",)),
            ("City", ("geonames",)),
            ("No_Country_Region", ("geonames",)),
            ("ClimateZone", ("gcmd",)),
        ],
    },
]

NODE_CLUSTER = {n: c["key"] for c in CLUSTERS for n, _ in c["nodes"]}
NODE_PROVENANCE = {n: p for c in CLUSTERS for n, p in c["nodes"]}

# The five geographic node types that every "region" relationship fans out to. In the
# diagram these are bundled into a single edge clipped at the Geography cluster border.
REGION_TYPES = ["Continent", "Country", "Country_Subdivision", "City", "No_Country_Region"]

# --------------------------------------------------------------------------------------
# Edges. (source, relation, target). `target` may be "@geo" to mean "bundled edge into
# every REGION_TYPES node", or "@mentions" for the PAPER_MENTIONS fan-out.
# --------------------------------------------------------------------------------------
EDGES = [
    # --- publication metadata -------------------------------------------------------
    ("Paper", "PAPER_AUTHORED_BY", "Author"),
    ("Paper", "PAPER_PUBLISHED_IN_VENUE", "Venue"),
    ("Paper", "PAPER_BELONGS_TO_FIELD", "Field"),
    ("Paper", "PAPER_HAS_KEYWORD", "Keyword"),
    ("Paper", "CITED_BY", "Paper"),
    # --- paper -> research content ---------------------------------------------------
    ("Paper", "PAPER_WORKS_ON_TASK", "Task"),
    ("Paper", "PAPER_SOLVES_PROBLEM", "Problem"),
    ("Paper", "PAPER_APPLIES_METHOD", "Method"),
    ("Paper", "PAPER_HAS_RESULT", "Result"),
    ("Paper", "PAPER_HAS_INNOVATION", "Innovation"),
    ("Paper", "PAPER_USES_METRIC", "EvaluationMetric"),
    ("Paper", "PAPER_EXPERIMENTS_ON_OBSERVATIONAL_DATASET", "ObservationalDataset"),
    ("Paper", "PAPER_USES_MODEL", "Source"),
    # --- research content internal ---------------------------------------------------
    ("Task", "TASK_FACES_PROBLEM", "Problem"),
    ("Task", "TASK_USES_METRIC", "EvaluationMetric"),
    ("Task", "TASK_EXPERIMENTS_ON_OBSERVATIONAL_DATASET", "ObservationalDataset"),
    ("Method", "METHOD_SOLVES_PROBLEM", "Problem"),
    ("Method", "METHOD_WORKS_ON_TASK", "Task"),
    ("Method", "METHOD_HAS_RESULT", "Result"),
    ("Method", "METHOD_HAS_INNOVATION", "Innovation"),
    ("Method", "METHOD_USES_METRIC", "EvaluationMetric"),
    ("Method", "METHOD_EXPERIMENTS_ON_OBSERVATIONAL_DATASET", "ObservationalDataset"),
    ("Result", "RESULT_HAS_METRIC", "EvaluationMetric"),
    # --- model (Source) as an evaluated artefact --------------------------------------
    ("Source", "MODEL_HAS_RESULT", "Result"),
    ("Source", "MODEL_USES_METRIC", "EvaluationMetric"),
    ("Source", "MODEL_EXPERIMENTS_ON_OBSERVATIONAL_DATASET", "ObservationalDataset"),
    # --- ESGF core -------------------------------------------------------------------
    ("Source", "IS_OF_TYPE", "SourceType"),
    ("Source", "HAS_SOURCE_COMPONENT", "SourceComponent"),
    ("Source", "HAS_SUBSEQUENT_VERSION", "Source"),
    ("Source", "INHERITED_FROM", "Source"),
    ("Source", "PRODUCES_VARIABLE", "Variable"),
    ("Source", "APPLIES_TO_REALM", "Realm"),
    ("Source", "USED_IN_EXPERIMENT", "Experiment"),
    ("Source", "GENERATED_BY_ACTIVITY", "Activity"),
    ("Source", "PRODUCED_BY_INSTITUTE", "Institute"),
    ("Source", "BELONGS_TO_MIP_ERA", "MIPEra"),
    ("Source", "PART_OF_PROJECT", "Project"),
    ("Source", "ASSOCIATED_WITH_MEMBER", "Member"),
    ("Source", "PART_OF_ENSEMBLE", "Ensemble"),
    ("Source", "SAMPLED_AT_FREQUENCY", "Frequency"),
    ("Source", "HAS_SPATIAL_RESOLUTION", "Resolution"),
    ("SourceComponent", "HAS_SOURCE_COMPONENT", "SourceComponent"),
    ("SourceComponent", "HAS_SUBSEQUENT_VERSION", "SourceComponent"),
    ("SourceComponent", "PRODUCES_VARIABLE", "Variable"),
    ("SourceComponent", "APPLIES_TO_REALM", "Realm"),
    ("Variable", "HAS_SPATIAL_RESOLUTION", "Resolution"),
    ("Experiment", "HAS_SUB_EXPERIMENT", "SubExperiment"),
    ("Experiment", "APPLIES_TO_REALM", "Realm"),
    ("Experiment", "USES_FORCING", "Forcing"),
    ("Experiment", "INCLUDES_ENSEMBLE_MEMBER", "Ensemble"),
    ("Experiment", "PERFORMED_BY_INSTITUTE", "Institute"),
    ("Experiment", "PART_OF_PROJECT", "Project"),
    ("Experiment", "HAS_SPATIAL_RESOLUTION", "Resolution"),
    ("ExperimentFamily", "INCLUDES_EXPERIMENT", "Experiment"),
    ("ExperimentFamily", "BELONGS_TO_MIP_ERA", "MIPEra"),
    ("SubExperiment", "PART_OF_ENSEMBLE", "Ensemble"),
    ("Activity", "FOCUSES_ON_REALM", "Realm"),
    ("Activity", "PART_OF_PROJECT", "Project"),
    ("Institute", "PARTICIPATED_IN", "Project"),
    ("Project", "COVERS_DOMAIN", "Domain"),
    ("RCM", "DRIVEN_BY_SOURCE", "Source"),
    ("RCM", "USED_IN_EXPERIMENT", "Experiment"),
    ("RCM", "PART_OF_PROJECT", "Project"),
    ("RCM", "COVERS_DOMAIN", "Domain"),
    ("RCM", "PRODUCES_VARIABLE", "Variable"),
    ("RCM", "HAS_SPATIAL_RESOLUTION", "Resolution"),
    ("RCM", "COVERS_REGION", "@geo"),
    # --- model characteristics --------------------------------------------------------
    ("Source", "USES_PHYSICAL_SCHEME", "PhysicalScheme"),
    ("Source", "HAS_PHYSICAL_FEATURE", "PhysicalFeature"),
    ("Source", "HAS_SIMULATION_TYPE", "SimulationType"),
    ("Source", "HAS_METRIC", "Metric"),
    # --- phenomena & observation ------------------------------------------------------
    ("Instrument", "MEASURES_VARIABLE", "Variable"),
    ("Instrument", "MOUNTED_ON", "Platform"),
    ("Natural_Hazard", "IS_OF_TYPE", "NaturalHazardType"),
    ("Natural_Hazard", "AFFECTED_REGION", "@geo"),
    ("Weather_Event", "IMPACTED_REGION", "@geo"),
    ("Teleconnection", "INFLUENCES_REGION", "@geo"),
    ("Ocean_Circulation", "OBSERVED_IN_REGION", "@geo"),
    # --- geography internal ------------------------------------------------------------
    ("Country", "IN_CONTINENT", "Continent"),
    ("Country_Subdivision", "IN_COUNTRY", "Country"),
    ("City", "IN_COUNTRY", "Country"),
    ("No_Country_Region", "HAS_CLIMATE_ZONE", "ClimateZone"),
]

# Node types that PAPER_MENTIONS links to. Drawn as three bundled edges into the ESGF,
# phenomena and geography clusters rather than ~35 individual arrows.
MENTION_TARGET_CLUSTERS = ["esgf", "modelchar", "phenom", "geo"]

# --------------------------------------------------------------------------------------
# DOT generation
# --------------------------------------------------------------------------------------

FONT = "Helvetica"

def _wrap_label(rel, max_len=22):
    """Break a long relationship name across two lines at an underscore near the middle."""
    if len(rel) <= max_len:
        return rel
    parts = rel.split("_")
    best, best_cost = None, None
    for i in range(1, len(parts)):
        left, right = "_".join(parts[:i]) + "_", "_".join(parts[i:])
        cost = abs(len(left) - len(right)) + max(len(left), len(right))
        if best_cost is None or cost < best_cost:
            best, best_cost = (left, right), cost
    return f"{best[0]}\\n{best[1]}"


def _fill(node):
    """Striped fill when a node type is populated from more than one source."""
    colors = [SOURCES[s][1] for s in NODE_PROVENANCE[node]]
    if len(colors) == 1:
        return f'fillcolor="{colors[0]}", style="filled,rounded"'
    return f'fillcolor="{":".join(colors)}", style="striped,rounded"'


def _cluster_anchor(cluster_key):
    """Invisible point node used as the endpoint of bundled (cluster-level) edges."""
    return f"__anchor_{cluster_key}"

# --------------------------------------------------------------------------------------
# Layout
#
# `dot` alone produces either an extremely tall (rankdir=LR) or extremely wide
# (rankdir=TB) drawing of this schema, and the force-directed engines turn the ESGF
# cluster into a hairball. So the layout is built in two stages:
#
#   1. each cluster is laid out on its own with `dot` (compact, few crossings);
#   2. the clusters are placed into a designed column arrangement, and the whole graph
#      is rendered with `neato -n2`, which keeps the given coordinates and only routes
#      the edges.
#
# `neato -n2` does not draw cluster boxes, so they are injected into the SVG afterwards
# (see `_add_cluster_boxes`).
# --------------------------------------------------------------------------------------

# Rows of the final arrangement; each entry is a row of clusters placed side by side.
# Reading order: publications on top, the ESGF core in the middle, the geographic and
# phenomenon vocabularies at the bottom.
LAYOUT_ROWS = [
    ["pub", "claim"],
    ["esgf"],
    ["geo", "phenom"],
]

# Clusters laid out *inside* another cluster's `dot` run, so their nodes sit next to the
# nodes they attach to instead of being connected by long cross-diagram edges.
NESTED_CLUSTERS = {"esgf": ["modelchar"]}

# rankdir used for each cluster's internal `dot` layout, chosen so the cluster's aspect
# ratio suits its slot (run with --print-sizes to see both options).
CLUSTER_RANKDIR = {
    "pub": "LR",
    "claim": "LR",
    "esgf": "TB",
    "modelchar": "LR",
    "phenom": "LR",
    "geo": "LR",
}

# Where bundled (cluster-level) edges terminate on each cluster's box.
BUNDLE_ANCHOR_SIDE = {
    "esgf": "top",
    "modelchar": "top",
    "phenom": "top",
    "geo": "right",
}

COLUMN_GAP = 3.2      # inches between clusters in the same row
ROW_GAP = 2.0         # inches between rows
CLUSTER_PAD = 0.42    # inches of padding between a cluster's nodes and its box
CLUSTER_LABEL_PAD = 0.34  # extra padding at the top of a cluster box for its title

NODE_DEFAULTS = (
    f'  node [shape=box, style="filled,rounded", fontname="{FONT}-Bold", fontsize=15,\n'
    '        color="#333333", penwidth=1.2, margin="0.16,0.08", height=0.34];'
)
EDGE_DEFAULTS = (
    f'  edge [fontname="{FONT}", fontsize=10.5, color="#6b6b6b", fontcolor="#2f2f2f",\n'
    "        arrowsize=0.75, penwidth=1.0];"
)


def _plain_layout(source, engine="dot"):
    """Run an engine with -Tplain and parse the result.

    Returns (nodes, edges, size) in inches, where `nodes` maps a name to (x, y, w, h)
    and `edges` is a list of (tail, head, spline_points, label, label_pos).
    """
    out = subprocess.run(
        [engine, "-Tplain"], input=source, capture_output=True, text=True, check=True
    ).stdout
    nodes, edges, size = {}, [], (0.0, 0.0)
    for line in out.splitlines():
        parts = shlex.split(line)
        if not parts:
            continue
        if parts[0] == "graph":
            size = (float(parts[2]), float(parts[3]))
        elif parts[0] == "node":
            nodes[parts[1]] = tuple(float(v) for v in parts[2:6])
        elif parts[0] == "edge":
            count = int(parts[3])
            points = [
                (float(parts[4 + 2 * i]), float(parts[5 + 2 * i])) for i in range(count)
            ]
            rest = parts[4 + 2 * count :]
            # a labelled edge is followed by "<label> <xl> <yl>" before style/colour
            if len(rest) > 2:
                label, label_pos = rest[0], (float(rest[1]), float(rest[2]))
            else:
                label, label_pos = None, None
            edges.append((parts[1], parts[2], points, label, label_pos))
    return nodes, edges, size


def _intra_cluster_edges(node_names):
    return [(s, r, t) for s, r, t in EDGES if s in node_names and t in node_names]


def _cluster_layout(cluster):
    """Lay out a cluster (with any nested clusters) using `dot`.

    Returns (nodes, edges, size) in the cluster's own coordinate system, in inches.
    """
    by_key = {c["key"]: c for c in CLUSTERS}
    nested_keys = NESTED_CLUSTERS.get(cluster["key"], [])
    names = [n for n, _ in cluster["nodes"]]
    for key in nested_keys:
        names += [n for n, _ in by_key[key]["nodes"]]

    lines = [
        "digraph c {",
        "  compound=true;",
        f'  rankdir={CLUSTER_RANKDIR[cluster["key"]]};',
        "  nodesep=0.30; ranksep=0.85; splines=spline;",
        NODE_DEFAULTS,
        EDGE_DEFAULTS,
    ]
    for name, _ in cluster["nodes"]:
        lines.append(f'  "{name}" [{_fill(name)}];')
    for key in nested_keys:
        nested = by_key[key]
        lines.append(f"  subgraph cluster_{key} {{")
        lines.append(f'    label="{nested["label"]}"; labelloc=t;')
        lines.append(f'    fontname="{FONT}-Bold"; fontsize=19; margin=20;')
        for name, _ in nested["nodes"]:
            lines.append(f'    "{name}" [{_fill(name)}];')
        lines.append("  }")
    # `dot` places every self-loop on a node at the same height, so their labels collide;
    # blank leading lines push each successive one further down.
    self_loop_seen = defaultdict(int)
    for src, rel, dst in _intra_cluster_edges(names):
        label = _wrap_label(rel)
        if src == dst:
            label = "\\n" * (2 * self_loop_seen[src]) + label
            self_loop_seen[src] += 1
        lines.append(f'  "{src}" -> "{dst}" [label="{label}"];')
    lines.append("}")
    return _plain_layout("\n".join(lines))


def _label_min_width(text, font_size=19):
    """Rough width in inches of a bold Helvetica cluster title, so boxes fit their label."""
    return (len(text) * font_size * 0.56 + 34) / 72


def compute_layout():
    """Lay out every cluster, then place the clusters into LAYOUT_ROWS.

    Returns (node_positions_in_inches, cluster_boxes) where a cluster box is
    (key, x0, y0, x1, y1) covering the padded extent of the cluster.
    """
    nested_of = {n: parent for parent, keys in NESTED_CLUSTERS.items() for n in keys}
    laid_out = {}
    for cluster in CLUSTERS:
        if cluster["key"] in nested_of:
            continue  # laid out together with its parent
        positions, edges, size = _cluster_layout(cluster)
        # box of the cluster including padding, room for its title, and enough width
        # for the title text itself
        width = max(size[0] + 2 * CLUSTER_PAD, _label_min_width(cluster["label"]))
        laid_out[cluster["key"]] = {
            "positions": positions,
            "edges": edges,
            "width": width,
            "x_offset": (width - size[0]) / 2,
            "height": size[1] + 2 * CLUSTER_PAD + CLUSTER_LABEL_PAD,
        }

    row_widths = [
        sum(laid_out[key]["width"] for key in row) + COLUMN_GAP * (len(row) - 1)
        for row in LAYOUT_ROWS
    ]
    row_heights = [max(laid_out[key]["height"] for key in row) for row in LAYOUT_ROWS]
    total_width = max(row_widths)
    total_height = sum(row_heights) + ROW_GAP * (len(LAYOUT_ROWS) - 1)

    node_positions, boxes, routed_edges = {}, [], []
    y_top = total_height
    for row, row_width, row_height in zip(LAYOUT_ROWS, row_widths, row_heights):
        x_cursor = (total_width - row_width) / 2  # centre the row horizontally
        for key in row:
            entry = laid_out[key]
            # centre the cluster vertically within its row
            y0 = y_top - row_height + (row_height - entry["height"]) / 2
            x0 = x_cursor
            boxes.append((key, x0, y0, x0 + entry["width"], y0 + entry["height"]))
            # coordinates from the per-cluster layout are relative to its own origin
            dx = x0 + entry["x_offset"]
            dy = y0 + CLUSTER_PAD
            for name, (nx, ny, _w, _h) in entry["positions"].items():
                node_positions[name] = (dx + nx, dy + ny)
            # boxes for clusters nested inside this one, sized to their member nodes
            for nested_key in NESTED_CLUSTERS.get(key, []):
                members = [n for n, _ in next(c for c in CLUSTERS if c["key"] == nested_key)["nodes"]]
                xs = [node_positions[n][0] for n in members]
                ys = [node_positions[n][1] for n in members]
                widths = [entry["positions"][n][2] for n in members]
                heights = [entry["positions"][n][3] for n in members]
                pad = CLUSTER_PAD * 0.6
                nx0 = min(x - w / 2 for x, w in zip(xs, widths)) - pad
                nx1 = max(x + w / 2 for x, w in zip(xs, widths)) + pad
                ny0 = min(y - h / 2 for y, h in zip(ys, heights)) - pad
                ny1 = max(y + h / 2 for y, h in zip(ys, heights)) + pad + CLUSTER_LABEL_PAD
                nested_box = next(c for c in CLUSTERS if c["key"] == nested_key)
                nx1 = max(nx1, nx0 + _label_min_width(nested_box["label"]))
                boxes.append((nested_key, nx0, ny0, nx1, ny1))
            for tail, head, points, label, label_pos in entry["edges"]:
                routed_edges.append(
                    (
                        tail,
                        head,
                        [(px + dx, py + dy) for px, py in points],
                        label,
                        None if label_pos is None else (label_pos[0] + dx, label_pos[1] + dy),
                    )
                )
            x_cursor += entry["width"] + COLUMN_GAP
        y_top -= row_height + ROW_GAP

    return node_positions, boxes, routed_edges


def _bundle_anchor(cluster_key, boxes):
    """Position for a bundled edge's endpoint: just outside a cluster's box edge."""
    _, x0, y0, x1, y1 = next(b for b in boxes if b[0] == cluster_key)
    side = BUNDLE_ANCHOR_SIDE.get(cluster_key, "left")
    if side == "top":
        # three-quarters along the top edge, clear of the cluster's title text
        return (x0 + 0.75 * (x1 - x0), y1 + 0.04)
    if side == "right":
        return (x1 + 0.04, (y0 + y1) / 2)
    return (x0 - 0.04, (y0 + y1) / 2)


def build_dot(title="ClimatePub4KG ontology", node_positions=None, boxes=None, routed_edges=None):
    """Emit the positioned graph for `neato -n2`."""
    if node_positions is None:
        node_positions, boxes, routed_edges = compute_layout()

    lines = []
    add = lines.append
    add("digraph ontology {")
    add("  splines=true;")
    add("  outputorder=edgesfirst;")
    add('  bgcolor="white";')
    add("  pad=0.1;")
    add(NODE_DEFAULTS)
    add(EDGE_DEFAULTS)
    add("")

    def pt(pos):
        return f"{pos[0] * 72:.2f},{pos[1] * 72:.2f}"

    # ---- nodes -----------------------------------------------------------------------
    for cluster in CLUSTERS:
        add(f'  // {cluster["label"]}')
        for name, _ in cluster["nodes"]:
            add(f'  "{name}" [{_fill(name)}, pos="{pt(node_positions[name])}"];')
    add("")

    # ---- invisible spacers so the drawing's bounding box covers the cluster boxes ------
    add("  // spacers keeping the cluster boxes inside the drawing")
    for i, (_, x0, y0, x1, y1) in enumerate(boxes):
        for j, (x, y) in enumerate([(x0, y0), (x1, y1)]):
            add(f'  "__sp{i}_{j}" [shape=point, width=0.01, style=invis, pos="{pt((x, y))}"];')
    add("")

    # ---- bundled endpoints -------------------------------------------------------------
    for cluster in CLUSTERS:
        key = cluster["key"]
        pos = _bundle_anchor(key, boxes)
        add(f'  "{_cluster_anchor(key)}" [shape=point, width=0.01, style=invis, pos="{pt(pos)}"];')
    add("")

    # ---- intra-cluster edges: reuse the splines and label positions `dot` computed ------
    add("  // relationships inside a cluster (routed by dot)")
    for tail, head, points, label, label_pos in routed_edges:
        spline = "e," + pt(points[-1]) + " " + " ".join(pt(p) for p in points)
        attrs = [f'pos="{spline}"']
        if label is not None:
            attrs.append(f'label="{label}", lp="{pt(label_pos)}"')
        add(f'  "{tail}" -> "{head}" [{", ".join(attrs)}];')
    add("")

    # ---- edges crossing clusters: routed by neato ---------------------------------------
    add("  // relationships crossing clusters")
    intra = {(t, h) for t, h, *_ in routed_edges}
    for src, rel, dst in EDGES:
        if dst == "@geo":
            add(
                f'  "{src}" -> "{_cluster_anchor("geo")}" '
                f'[label="{rel}", penwidth=1.8, color="#4c7a1f", '
                'fontcolor="#38590f", arrowsize=1.0, fontsize=12];'
            )
        elif (src, dst) not in intra:
            add(f'  "{src}" -> "{dst}" [label="{_wrap_label(rel)}"];')
    add("")

    add("  // PAPER_MENTIONS: Paper -> every domain entity type")
    for i, key in enumerate(MENTION_TARGET_CLUSTERS):
        label = "PAPER_MENTIONS" if i == 0 else ""
        add(
            f'  "Paper" -> "{_cluster_anchor(key)}" [label="{label}", '
            'style=dashed, penwidth=2.4, color="#7b2d8b", fontcolor="#7b2d8b", '
            "fontsize=13, arrowsize=1.1];"
        )
    add("}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# SVG post-processing: cluster boxes, title and legend
# --------------------------------------------------------------------------------------

def _svg_escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


TITLE_BAND = 96     # points of extra canvas at the top for the title
LEGEND_BAND = 270   # points of extra canvas at the bottom for the legend


def _expand_canvas(svg, top, bottom):
    """Grow the graphviz SVG canvas, shift its contents down and repaint the background."""
    width = float(re.search(r'<svg width="([\d.]+)pt"', svg).group(1))
    height = float(re.search(r'height="([\d.]+)pt"', svg).group(1))
    vx, vy, vw, vh = (
        float(v) for v in re.search(r'viewBox="([\d.\- ]+)"', svg).group(1).split()
    )

    svg = re.sub(
        r'<svg width="[\d.]+pt" height="[\d.]+pt"',
        f'<svg width="{width:.0f}pt" height="{height + top + bottom:.0f}pt"',
        svg,
        count=1,
    )
    svg = re.sub(
        r'viewBox="[\d.\- ]+"', f'viewBox="{vx} {vy} {vw} {vh + top + bottom}"', svg, count=1
    )
    # push the drawing down so the freed space ends up above it
    svg = re.sub(
        r'(<g id="graph0"[^>]*translate\()([\d.\-]+) ([\d.\-]+)(\))',
        lambda m: f"{m.group(1)}{m.group(2)} {float(m.group(3)) + top}{m.group(4)}",
        svg,
        count=1,
    )
    # a white page behind everything (graphviz's own background only covers the old box)
    background = (
        f'<rect x="{vx}" y="{vy}" width="{vw}" height="{vh + top + bottom}" fill="white"/>'
    )
    insert_at = svg.index(">", svg.index("<svg ")) + 1
    return svg[:insert_at] + "\n" + background + svg[insert_at:]


def decorate_svg(svg, boxes, title):
    """Insert cluster rectangles (behind everything), the title and the legend."""
    cluster_by_key = {c["key"]: c for c in CLUSTERS}
    svg = _expand_canvas(svg, TITLE_BAND, LEGEND_BAND)

    box_svg = ['<g class="clusters">']
    for key, x0, y0, x1, y1 in boxes:
        cluster = cluster_by_key[key]
        x, y = x0 * 72, -y1 * 72
        w, h = (x1 - x0) * 72, (y1 - y0) * 72
        box_svg.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="14" ry="14" '
            f'fill="{cluster["bg"]}" stroke="{cluster["color"]}" stroke-width="2.2"/>'
        )
        box_svg.append(
            f'<text x="{x + 16:.2f}" y="{y + 27:.2f}" font-family="{FONT},sans-serif" '
            f'font-size="19" font-weight="bold" fill="{cluster["color"]}">'
            f'{_svg_escape(cluster["label"])}</text>'
        )
    box_svg.append("</g>")

    # place the boxes immediately after the white page background so they sit behind
    # the edges and nodes
    marker = "</title>\n"
    index = svg.index(marker) + len(marker)
    end_of_bg = svg.index("/>", svg.index("<polygon", index)) + 2
    svg = svg[:end_of_bg] + "\n" + "\n".join(box_svg) + svg[end_of_bg:]

    # ---- title and legend, positioned in the graph's own coordinate system -------------
    min_x = min(b[1] for b in boxes) * 72
    max_x = max(b[3] for b in boxes) * 72
    max_y = -max(b[4] for b in boxes) * 72
    min_y = -min(b[2] for b in boxes) * 72

    overlay = ['<g class="overlay">']
    overlay.append(
        f'<text x="{(min_x + max_x) / 2:.2f}" y="{max_y - 46:.2f}" text-anchor="middle" '
        f'font-family="{FONT},sans-serif" font-size="40" font-weight="bold" fill="#1a1a1a">'
        f"{_svg_escape(title)}</text>"
    )

    lx, ly = min_x, min_y + 46
    rows = list(SOURCES.values()) + [("striped fill = populated from more than one source", None)]
    lw, lh = 430, 26 * len(rows) + 40
    overlay.append(
        f'<rect x="{lx:.2f}" y="{ly:.2f}" width="{lw}" height="{lh}" rx="10" ry="10" '
        'fill="#fafafa" stroke="#9a9a9a" stroke-width="1.6"/>'
    )
    overlay.append(
        f'<text x="{lx + 14:.2f}" y="{ly + 24:.2f}" font-family="{FONT},sans-serif" '
        'font-size="16" font-weight="bold" fill="#3d3d3d">Provenance of node instances</text>'
    )
    for i, (desc, color) in enumerate(rows):
        ry = ly + 40 + i * 26
        if color is None:
            swatch = (
                f'<rect x="{lx + 14:.2f}" y="{ry:.2f}" width="26" height="16" rx="3" '
                f'fill="{SOURCES["esgf"][1]}"/>'
                f'<rect x="{lx + 27:.2f}" y="{ry:.2f}" width="13" height="16" rx="3" '
                f'fill="{SOURCES["paper"][1]}"/>'
            )
        else:
            swatch = (
                f'<rect x="{lx + 14:.2f}" y="{ry:.2f}" width="26" height="16" rx="3" '
                f'fill="{color}" stroke="#666" stroke-width="0.8"/>'
            )
        overlay.append(swatch)
        overlay.append(
            f'<text x="{lx + 50:.2f}" y="{ry + 13:.2f}" font-family="{FONT},sans-serif" '
            f'font-size="14" fill="#2f2f2f">{_svg_escape(desc)}</text>'
        )
    overlay.append("</g>")

    insert_at = svg.rindex("</g>")
    return svg[:insert_at] + "\n".join(overlay) + "\n" + svg[insert_at:]


# --------------------------------------------------------------------------------------
# Verification against the live graph / declared ontology
# --------------------------------------------------------------------------------------

def verify(uri, user, password, database="neo4j"):
    from neo4j import GraphDatabase

    drawn = {(s, r, t) for s, r, t in EDGES if t != "@geo"}
    for s, r, t in EDGES:
        if t == "@geo":
            drawn.update((s, r, region) for region in REGION_TYPES)
    drawn_nodes = set(NODE_CLUSTER)

    # consts.py stores tuples as (source, target, relation)
    declared = {(s, r, t) for s, t, r in ALL_RELATIONSHIP_SOURCE_TO_TARGET}

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        live = set()
        mentions = defaultdict(int)
        query = (
            "MATCH (a)-[r]->(b) "
            "RETURN labels(a)[0] AS s, type(r) AS rel, labels(b)[0] AS t, count(*) AS c"
        )
        for row in session.run(query):
            if row["s"] is None or row["t"] is None or str(row["s"]).startswith("_"):
                continue
            if row["rel"] == "PAPER_MENTIONS":
                mentions[row["t"]] += row["c"]
                continue
            live.add((row["s"], row["rel"], row["t"]))
        live_nodes = {
            r["label"]
            for r in session.run("CALL db.labels() YIELD label RETURN label")
            if not r["label"].startswith("_")
        }
    driver.close()

    def report(name, items):
        print(f"\n{name} ({len(items)}):")
        if not items:
            print("    none")
            return
        for item in sorted(items):
            if isinstance(item, str):
                print(f"    {item}")
            else:
                print(f"    ({item[0]})-[:{item[1]}]->({item[2]})")

    print("=" * 78)
    print(f"drawn edges: {len(drawn)}   live edges: {len(live)}   declared in consts.py: {len(declared)}")
    report("live in Neo4j but NOT drawn", live - drawn)
    report("drawn but NOT present in Neo4j (declared-only)", drawn - live)
    report("declared in consts.py but NOT drawn", declared - drawn)
    report("node labels in Neo4j but NOT drawn", live_nodes - drawn_nodes)
    report("node types drawn but NOT in Neo4j", drawn_nodes - live_nodes)
    print(f"\nPAPER_MENTIONS target types in Neo4j: {len(mentions)}")
    missing_mentions = {t for t in mentions if NODE_CLUSTER.get(t) not in MENTION_TARGET_CLUSTERS}
    print(f"  targets outside the bundled clusters: {sorted(missing_mentions) or 'none'}")
    print("=" * 78)


# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--basename", default="climatepub4kg_ontology")
    parser.add_argument("--png-width", type=int, default=4096, help="width of the rendered PNG")
    parser.add_argument("--title", default="ClimatePub4KG ontology")
    parser.add_argument("--verify", action="store_true", help="diff the drawn ontology against live Neo4j")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:8687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "neo4j"))
    args = parser.parse_args()

    if args.verify:
        verify(args.neo4j_uri, args.neo4j_user, args.neo4j_password)

    os.makedirs(args.output_dir, exist_ok=True)

    if shutil.which("dot") is None or shutil.which("neato") is None:
        raise SystemExit("graphviz (`dot` and `neato`) must be on PATH")

    node_positions, boxes, routed_edges = compute_layout()
    dot_source = build_dot(args.title, node_positions, boxes, routed_edges)

    dot_path = os.path.join(args.output_dir, f"{args.basename}.dot")
    with open(dot_path, "w") as handle:
        handle.write(dot_source)
    print(f"wrote {dot_path}")

    # `-n2` keeps the positions computed above and only routes the edges
    svg = subprocess.run(
        ["neato", "-n2", "-Tsvg"], input=dot_source, capture_output=True, text=True, check=True
    ).stdout
    svg = decorate_svg(svg, boxes, args.title)

    svg_path = os.path.join(args.output_dir, f"{args.basename}.svg")
    with open(svg_path, "w") as handle:
        handle.write(svg)
    print(f"wrote {svg_path}")

    import cairosvg

    png_path = os.path.join(args.output_dir, f"{args.basename}.png")
    cairosvg.svg2png(bytestring=svg.encode(), write_to=png_path, output_width=args.png_width)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
