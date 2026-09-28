"""
Parse date-partitioned PDFs with local olmOCR.

Default input:
  papers/pdfs/<date>/*.pdf

Default output:
  papers/parsed/<date>/*.md
"""

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OLMOCR_DIR = Path.home() / "olmocr"


def today() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d")


def latest_pdf_date() -> str:
    pdf_root = SCRIPT_DIR / "papers" / "pdfs"
    if not pdf_root.exists():
        return today()

    date_dirs = sorted(path.name for path in pdf_root.iterdir() if path.is_dir())
    if not date_dirs:
        return today()
    return date_dirs[-1]


def default_pdf_dir(date: str) -> Path:
    return SCRIPT_DIR / "papers" / "pdfs" / date


def default_output_dir(date: str) -> Path:
    return SCRIPT_DIR / "papers" / "parsed" / date


def default_workspace_dir(date: str) -> Path:
    return SCRIPT_DIR / "output" / "olmocr_workspaces" / date


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def iter_pdf_files(pdf_dir: Path) -> list[Path]:
    return sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())


def source_ref_for_olmocr(pdf_path: Path, cwd: Path) -> str:
    resolved_pdf = pdf_path.resolve()
    resolved_cwd = cwd.resolve()
    try:
        return resolved_pdf.relative_to(resolved_cwd).as_posix()
    except ValueError:
        return resolved_pdf.as_posix()


def markdown_path_for_source(workspace: Path, source_ref: str) -> Path:
    relative_path = source_ref.lstrip("/")
    safe_parts = [part for part in relative_path.split("/") if part and part != ".."]
    safe_relative_path = Path(*safe_parts) if safe_parts else Path(source_ref).name
    return workspace / "markdown" / safe_relative_path.with_suffix(".md")


def write_pdf_list(pdf_refs: list[str], workspace: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    list_path = workspace / "pdfs_to_parse.txt"
    with list_path.open("w", encoding="utf-8") as f:
        for pdf_ref in pdf_refs:
            f.write(f"{pdf_ref}\n")
    return list_path


def run_olmocr(
    pdf_refs: list[str],
    workspace: Path,
    olmocr_dir: Path,
    python_bin: str,
    model: str,
    workers: int,
    server: str | None,
    api_key: str | None,
    extra_args: list[str],
) -> None:
    if not olmocr_dir.exists():
        raise FileNotFoundError(f"olmOCR directory not found: {olmocr_dir}")

    pdf_list_path = write_pdf_list(pdf_refs, workspace)
    cmd = [
        python_bin,
        "-m",
        "olmocr.pipeline",
        str(workspace),
        "--markdown",
        # "--model",
        # model,
        "--workers",
        str(workers),
    ]

    if server:
        cmd.extend(["--server", server])
    if api_key:
        cmd.extend(["--api_key", api_key])

    cmd.extend(extra_args)
    cmd.extend(["--pdfs", str(pdf_list_path)])

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{olmocr_dir}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(olmocr_dir)
    )

    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)


def copy_markdown_outputs(
    pdfs: list[Path],
    pdf_refs: list[str],
    workspace: Path,
    output_dir: Path,
    overwrite: bool,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0

    for pdf_path, pdf_ref in zip(pdfs, pdf_refs):
        parsed_path = output_dir / f"{pdf_path.stem}.md"
        if parsed_path.exists() and not overwrite:
            continue

        markdown_path = markdown_path_for_source(workspace, pdf_ref)
        if not markdown_path.exists():
            print(f"Missing parsed markdown for {pdf_path.name}: {markdown_path}")
            missing += 1
            continue

        shutil.copyfile(markdown_path, parsed_path)
        copied += 1

    return copied, missing


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    default_date = latest_pdf_date()
    parser = argparse.ArgumentParser(
        description=(
            "Run local olmOCR for papers/pdfs/<date> and save parsed markdown "
            "to papers/parsed/<date>."
        )
    )
    parser.add_argument(
        "--date",
        default=default_date,
        help="PDF date partition to process. Defaults to latest papers/pdfs/<date>.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        help="Directory containing PDFs. Defaults to papers/pdfs/<date>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for parsed markdown. Defaults to papers/parsed/<date>.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="olmOCR workspace. Defaults to output/olmocr_workspaces/<date>.",
    )
    parser.add_argument(
        "--olmocr-dir",
        type=Path,
        default=DEFAULT_OLMOCR_DIR,
        help="Local olmOCR repository directory. Defaults to ~/olmocr.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run olmocr.pipeline.",
    )
    parser.add_argument(
        "--model",
        default="allenai/olmOCR-2-7B-1025-FP8",
        help="olmOCR model argument.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of olmOCR workers.",
    )
    parser.add_argument(
        "--server",
        help="Optional OpenAI-compatible vLLM server URL. If omitted, olmOCR starts local vLLM.",
    )
    parser.add_argument(
        "--api-key",
        help="Optional API key for an external server.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing parsed markdown files.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, extra_args = parse_args()
    pdf_dir = resolve_path(args.pdf_dir or default_pdf_dir(args.date))
    output_dir = resolve_path(args.output_dir or default_output_dir(args.date))
    workspace = resolve_path(args.workspace or default_workspace_dir(args.date))
    olmocr_dir = resolve_path(args.olmocr_dir)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdfs = iter_pdf_files(pdf_dir)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")

    selected_pdfs = [
        pdf_path
        for pdf_path in pdfs
        if args.force or not (output_dir / f"{pdf_path.stem}.md").exists()
    ]
    skipped_existing = len(pdfs) - len(selected_pdfs)

    if selected_pdfs:
        pdf_refs = [source_ref_for_olmocr(pdf_path, SCRIPT_DIR) for pdf_path in selected_pdfs]
        run_olmocr(
            pdf_refs=pdf_refs,
            workspace=workspace,
            olmocr_dir=olmocr_dir,
            python_bin=args.python,
            model=args.model,
            workers=args.workers,
            server=args.server,
            api_key=args.api_key,
            extra_args=extra_args,
        )
        copied, missing = copy_markdown_outputs(
            pdfs=selected_pdfs,
            pdf_refs=pdf_refs,
            workspace=workspace,
            output_dir=output_dir,
            overwrite=args.force,
        )
    else:
        copied = 0
        missing = 0

    print("\nolmOCR Parse Summary")
    print(f"  PDF dir: {pdf_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Workspace: {workspace}")
    print(f"  total_pdfs: {len(pdfs)}")
    print(f"  submitted: {len(selected_pdfs)}")
    print(f"  skipped_existing: {skipped_existing}")
    print(f"  copied_markdown: {copied}")
    print(f"  missing_markdown: {missing}")


if __name__ == "__main__":
    main()
