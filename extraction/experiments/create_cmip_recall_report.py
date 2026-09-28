#!/usr/bin/env python3
"""Create a paper-ready PDF report for CMIP recall evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVALUATION_DIR = (
    SCRIPT_DIR
    / "output"
    / "cmip_moreinfo_entity_evaluation"
    / "20260612_101052_cc460532_eval"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / "cmip_recall_report"
DEFAULT_MOREINFO_RESULT_PATH = (
    SCRIPT_DIR
    / "output"
    / "cmip_publication_moreinfo_entities"
    / "20260612_101052_cc460532"
    / "result.json"
)
PRESENT_ONLY_RECALL_PATH = (
    Path("parsed_ground_truth_presence") / "recall_ground_truth_present_only.json"
)

DISPLAY_NAMES = {
    "activities": "Activity",
    "experiments": "Experiment",
    "frequencies": "Frequency",
    "sources": "Source",
    "micro": "Micro Avg",
}

ENTITY_ORDER = ["activities", "experiments", "frequencies", "sources", "micro"]


def now_run_id() -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(value, encoding="utf-8")
    os.replace(tmp_path, path)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def code(value: Any) -> str:
    return f"<code>{esc(value)}</code>"


def strong(value: Any) -> str:
    return f"<strong>{esc(value)}</strong>"


def list_items(values: list[str]) -> str:
    return "".join(f"<li>{value}</li>" for value in values)


def metric_row(entity_type: str, metrics: dict[str, Any]) -> str:
    record = metrics[entity_type]
    recall = record["recall_on_present_ground_truth"]
    return (
        "<tr>"
        f"<th>{esc(DISPLAY_NAMES[entity_type])}</th>"
        f"<td>{record['ground_truth_present_in_parsed']:,}</td>"
        f"<td>{record['correct_present_in_parsed']:,}</td>"
        f"<td>{record['missed_present_in_parsed']:,}</td>"
        f"<td>{pct(recall)}</td>"
        "</tr>"
    )


def callout(title: str, body: str) -> str:
    return (
        '<aside class="callout">'
        f"<h4>{esc(title)}</h4>"
        f"{body}"
        "</aside>"
    )


def count_year(records: list[dict[str, Any]], year: int) -> int:
    target = str(year)
    return sum(
        1
        for record in records
        if str(record.get("site_year") or record.get("publication_year")) == target
    )


def build_report_html(
    summary: dict[str, Any],
    present_only: dict[str, Any],
    evaluation_dir: Path,
    moreinfo_result: dict[str, Any] | None,
) -> str:
    metrics = present_only["metrics"]
    micro = metrics["micro"]
    moreinfo_summary = (moreinfo_result or {}).get("summary") or {}
    cmip_publication_count = int(
        moreinfo_summary.get(
            "matched_record_count",
            summary["paper_count"] + summary.get("excluded_without_ground_truth_count", 0),
        )
    )
    records_with_moreinfo = int(
        moreinfo_summary.get("records_with_any_moreinfo", summary["paper_count"])
    )
    records_without_moreinfo = int(
        moreinfo_summary.get(
            "records_without_moreinfo",
            cmip_publication_count - records_with_moreinfo,
        )
    )
    site_records = (moreinfo_result or {}).get("site_records") or []
    paper_records = (moreinfo_result or {}).get("paper_records") or []
    site_2025_count = count_year(site_records, 2025)
    site_2026_count = count_year(site_records, 2026)
    subset_2025_count = count_year(paper_records, 2025)
    subset_2026_count = count_year(paper_records, 2026)
    total_gt = (
        micro["ground_truth_present_in_parsed"]
        + micro["ground_truth_absent_from_parsed"]
    )
    retained_rate = (
        micro["ground_truth_present_in_parsed"] / total_gt if total_gt else 0.0
    )
    table_rows = "\n".join(metric_row(entity_type, metrics) for entity_type in ENTITY_ORDER)

    manual_selection_examples = callout(
        "Evidence of Near, but Not Exact, Manual Labels",
        (
            "<p>The diagnostics show cases where the CMIP-selected entity is close "
            "to, but not the same as, the label observed in the paper or extraction "
            "pipeline:</p>"
            "<ul>"
            f"<li>{code('CCSM4')} was paired with the broader label {code('ccsm')} for {code('W1966668438')}.</li>"
            f"<li>{code('MRI-AGCM3.2H')} was paired with {code('MRI-AGCM')} for {code('W1996567291')}.</li>"
            f"<li>{code('BCC-CSM1.1-m')} was paired with {code('BCC-CSM1.1')} for {code('W2020963742')}.</li>"
            f"<li>{code('CSIRO-Mk3.6.0')} was paired with {code('CSIRO MK3.6')} for {code('W2040216654')}.</li>"
            "</ul>"
            "<p>These examples are consistent with a manual checkbox workflow: "
            "the selected option may be a related model family, nearby variant, "
            "or formatting variant rather than the exact surface form in the paper.</p>"
        ),
    )
    paper_selection_example = callout(
        "Paper Selection Example",
        (
            "<p>OpenAlex paper "
            f"{code('W1975283529')}, DOI {code('10.1007/s11434-012-5533-0')}, "
            "was matched to CMIP Publication Hub site record "
            f"{code('387')} by DOI and title.</p>"
        ),
    )
    ground_truth_example = callout(
        "More Info Ground Truth Example",
        (
            "<p>For "
            f"{code('W1975283529')}, <strong>More Info</strong> lists experiments "
            f"{code('historical')} and {code('rcp45-cmip5')}, frequency "
            f"{code('Monthly')}, and sources including {code('BCC-CSM1.1')}, "
            f"{code('CanESM2')}, {code('CCSM4')}, {code('CNRM-CM5')}, "
            f"and {code('CSIRO-Mk3.6.0')}.</p>"
        ),
    )
    retention_examples = callout(
        "Parsed-Presence Examples",
        (
            "<ul>"
            f"<li>Substring match: {code('lgm')} was found as {code('LGM')}.</li>"
            f"<li>Flexible-token match: {code('abrupt-4xCO2')} was found as {code('abrupt4xCO2')}.</li>"
            f"<li>Compact alphanumeric match: {code('hist-GHG')} was found despite punctuation or spacing differences.</li>"
            f"<li>Absent ground truth: for {code('W1966668438')}, {code('midHolocene')}, "
            f"{code('CCSM4')}, and {code('Monthly')} were CMIP ground truth entities, "
            "but their surface forms were not found in the parsed paper text.</li>"
            "</ul>"
        ),
    )
    result_examples = callout(
        "Remaining Miss Examples",
        (
            "<p>The retained-denominator evaluation still exposes real recall "
            "misses where the entity was present in parsed text but not extracted:</p>"
            "<ul>"
            f"<li>Activity: {code('CMIP')} in {code('W3001504003')}.</li>"
            f"<li>Experiment: {code('lgm')} in {code('W1966668438')}.</li>"
            f"<li>Source: {code('HadCM3')} in {code('W2020963742')}.</li>"
            f"<li>Frequency: {code('Daily')} in {code('W1980905972')}.</li>"
            "</ul>"
        ),
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CMIP Recall Evaluation Report</title>
  <style>
    @page {{
      size: Letter;
      margin: 0.72in;
    }}

    body {{
      color: #17202a;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 10.5pt;
      line-height: 1.42;
      margin: 0;
    }}

    h1 {{
      color: #0f3157;
      font-size: 20pt;
      margin: 0 0 6px;
    }}

    h2 {{
      border-bottom: 1px solid #c8d2dc;
      color: #173b63;
      font-size: 13pt;
      margin: 20px 0 8px;
      padding-bottom: 3px;
    }}

    h3 {{
      color: #244862;
      font-size: 11pt;
      margin: 14px 0 5px;
    }}

    p {{
      margin: 6px 0;
    }}

    code {{
      background: #eef3f7;
      border: 1px solid #d9e2ea;
      border-radius: 3px;
      color: #0e3a5b;
      font-family: "DejaVu Sans Mono", Consolas, monospace;
      font-size: 9pt;
      padding: 1px 3px;
    }}

    .meta {{
      color: #53616e;
      font-size: 9pt;
      margin-bottom: 16px;
    }}

    .headline {{
      background: #edf6f3;
      border-left: 4px solid #2d7d68;
      margin: 14px 0;
      padding: 10px 12px;
    }}

    .headline strong {{
      color: #145743;
    }}

    table {{
      border-collapse: collapse;
      margin: 10px 0 8px;
      width: 100%;
    }}

    th, td {{
      border: 1px solid #cfd8e1;
      padding: 6px 8px;
      text-align: right;
      vertical-align: top;
    }}

    th {{
      background: #eef3f7;
      color: #173b63;
      font-weight: 700;
    }}

    tbody th {{
      text-align: left;
    }}

    .callout {{
      background: #fbfcfd;
      border: 1px solid #d9e2ea;
      border-left: 4px solid #608bad;
      border-radius: 4px;
      margin: 8px 0;
      padding: 8px 10px;
      page-break-inside: avoid;
    }}

    .callout h4 {{
      color: #173b63;
      font-size: 10pt;
      margin: 0 0 4px;
    }}

    ul {{
      margin: 5px 0 2px 18px;
      padding: 0;
    }}

    li {{
      margin: 3px 0;
    }}

    .footer {{
      border-top: 1px solid #c8d2dc;
      color: #53616e;
      font-size: 8.5pt;
      margin-top: 18px;
      padding-top: 7px;
    }}
  </style>
</head>
<body>
  <h1>Recall Evaluation Using CMIP Publication Metadata</h1>

  <section class="headline">
    <p>
      Only <strong>{pct(retained_rate)}</strong> of CMIP ground-truth entities
      appeared in the parsed paper text. After restricting evaluation to these
      recoverable entities, the method achieved
      <strong>{pct(micro['recall_on_present_ground_truth'])} micro-average recall</strong>.
    </p>
  </section>

  <h2>Problem Statement</h2>
  <p>
    The CMIP Publication Hub depends on manually submitted publication metadata.
    A user manually adds a publication and then selects relevant metadata from
    controlled checkbox lists, including projects, activities, experiments,
    frequencies, sources, realms, and variables. In this form-based curation
    process, the user first selects the project era, then chooses the relevant
    CMIP characteristics from long lists of allowed options.
  </p>
  <p>
    This manual process creates two practical problems. First, adding papers is
    tedious because the system does not discover and tag publications
    automatically. This may help explain the dwindling number of recent
    additions: in the scraped CMIP citation records, there were {site_2025_count}
    records from 2025 but only {site_2026_count} from 2026; in the matched
    OpenAlex/PDF subset, there were {subset_2025_count} records from 2025 and
    {subset_2026_count} from 2026. Second, the selected metadata is not always
    an exact reflection of the text in the paper. Curators may select a related
    model family, a nearby model variant, or a formatting variant when the exact
    entity mentioned in the paper is not obvious or not selected.
  </p>
  {manual_selection_examples}

  <h2>Experimental Setup</h2>
  <h3>Paper Selection</h3>
  <p>
    We first scraped the CMIP Publication Hub for paper DOIs. For those same
    DOIs, we queried OpenAlex and retained papers only when OpenAlex provided
    open-access metadata and a downloadable PDF. This produced the CMIP paper
    subset used for the recall evaluation. Among {cmip_publication_count}
    retained CMIP-publication papers in this workflow, only
    {records_with_moreinfo} had usable entity information in <strong>More Info</strong>;
    {records_without_moreinfo} papers had no extracted entity metadata and were
    excluded from the entity-level recall calculation. The final evaluation set
    therefore contains {summary['paper_count']:,} CMIP papers.
  </p>
  {paper_selection_example}

  <h3>Ground Truth Construction</h3>
  <p>
    For each retained paper, CMIP Publication Hub <strong>More Info</strong> metadata was used
    as the external ground truth. Although the submission form supports broader
    metadata such as projects, realms, and variables, the recall evaluation in
    this report uses the four CMIP More Info entity families available in our
    extracted ground truth: Activities, Experiments, Sources, and Frequencies.
  </p>
  {ground_truth_example}

  <h3>Parsed-Paper Retention Logic</h3>
  <p>
    Because CMIP website metadata can include entities that are not explicitly
    stated in the paper text, we checked whether each ground-truth entity
    appeared in the parsed Markdown paper. The check was applied to all
    ground-truth entities, not only false negatives.
  </p>
  <p>
    The parsed-presence check normalized Unicode, dash variants, quote variants,
    and whitespace. It then searched with three increasingly tolerant rules:
    case-insensitive surface-form substring matching, flexible token-sequence
    matching that tolerates punctuation and spacing differences, and compact
    alphanumeric matching.
  </p>
  {retention_examples}

  <h2>Results and Discussion</h2>
  <p>
    The table reports recall after removing ground-truth entities whose surface
    forms were not found in the parsed paper text. This changes the question
    from “did we reproduce every manually selected CMIP checkbox?” to “did we
    extract the CMIP entities that were actually recoverable from the paper?”.
  </p>
  <table>
    <thead>
      <tr>
        <th>Entity Type</th>
        <th>Ground Truth Present in Paper</th>
        <th>Correctly Extracted</th>
        <th>Missed</th>
        <th>Recall</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
  <p>
    The overall micro-average recall is
    <strong>{pct(micro['recall_on_present_ground_truth'])}</strong>, with
    {micro['correct_present_in_parsed']:,} correctly extracted entities out of
    {micro['ground_truth_present_in_parsed']:,} ground-truth entities present in
    the parsed papers. Source extraction is strongest at
    {pct(metrics['sources']['recall_on_present_ground_truth'])}, which matters
    because source entities make up the largest part of the retained ground
    truth. Experiments are also relatively strong at
    {pct(metrics['experiments']['recall_on_present_ground_truth'])}. Activities
    and frequencies remain weaker, at
    {pct(metrics['activities']['recall_on_present_ground_truth'])} and
    {pct(metrics['frequencies']['recall_on_present_ground_truth'])},
    respectively, indicating that generic or compact metadata labels still
    require targeted improvement.
  </p>
  {result_examples}

  <h2>Conclusion</h2>
  <p>
    The lower recall observed against the full CMIP metadata denominator is
    partly an artifact of evaluating against entities that are tagged on the
    CMIP website but absent from the parsed paper text. When recall is computed
    only over ground-truth entities that are actually present in the parsed
    papers, the method recovers {micro['correct_present_in_parsed']:,} of
    {micro['ground_truth_present_in_parsed']:,} entities, yielding a
    micro-average recall of {pct(micro['recall_on_present_ground_truth'])}.
  </p>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PDF report for CMIP More Info recall evaluation."
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=DEFAULT_EVALUATION_DIR,
        help="Directory containing CMIP More Info evaluation outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory under which a timestamped report directory is created.",
    )
    parser.add_argument(
        "--moreinfo-result",
        type=Path,
        default=DEFAULT_MOREINFO_RESULT_PATH,
        help="CMIP More Info extraction result JSON used for paper-selection context.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional report run id. Defaults to a timestamp plus short UUID.",
    )
    return parser.parse_args()


def render_pdf(html_path: Path, pdf_path: Path) -> None:
    try:
        from weasyprint import HTML
    except ImportError:
        executable = shutil.which("weasyprint")
        fallback = Path.home() / "miniconda3" / "bin" / "weasyprint"
        if executable is None and fallback.exists():
            executable = str(fallback)
        if executable is None:
            raise RuntimeError(
                "weasyprint is required to render the PDF. "
                f"HTML report was written to {html_path}."
            )
        subprocess.run(
            [executable, str(html_path), str(pdf_path)],
            check=True,
        )
        return

    HTML(filename=str(html_path)).write_pdf(str(pdf_path))


def main() -> int:
    args = parse_args()
    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_id = args.run_id or now_run_id()
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = evaluation_dir / "summary.json"
    present_only_path = evaluation_dir / PRESENT_ONLY_RECALL_PATH
    summary = read_json(summary_path)
    present_only = read_json(present_only_path)
    moreinfo_result_path = args.moreinfo_result.expanduser().resolve()
    moreinfo_result = (
        read_json(moreinfo_result_path) if moreinfo_result_path.exists() else None
    )

    html_report = build_report_html(
        summary,
        present_only,
        evaluation_dir,
        moreinfo_result,
    )
    html_path = output_dir / "cmip_recall_report.html"
    pdf_path = output_dir / "cmip_recall_report.pdf"
    write_text(html_path, html_report)

    render_pdf(html_path, pdf_path)

    print(f"HTML report: {html_path}")
    print(f"PDF report: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
