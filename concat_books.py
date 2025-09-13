import argparse
import sys
from pathlib import Path
import shutil

def collect_txt_files(input_dir, recursive):
    pattern = "**/*.txt" if recursive else "*.txt"
    files = sorted(input_dir.glob(pattern), key=lambda p: str(p).lower())
    return [p for p in files if p.is_file()]

def main():
    ap = argparse.ArgumentParser(description="Concatenate .txt files into one corpus.")
    ap.add_argument("--input", "-i", type=Path, default=Path("books"),
                    help="Input directory containing .txt files (default: books)")
    ap.add_argument("--output", "-o", type=Path, default=Path("corpus.txt"),
                    help="Output corpus file path (default: corpus.txt)")
    ap.add_argument("--recursive", "-r", action="store_true", default=True,
                    help="Recurse into subdirectories (default: on)")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false",
                    help="Disable recursion into subdirectories")
    ap.add_argument("--force", "-f", action="store_true",
                    help="Overwrite output file if it exists")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="Less verbose output")
    args = ap.parse_args()

    in_dir = args.input.resolve()
    out_path = args.output.resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[error] Input directory not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    if out_path.exists() and not args.force:
        print(f"[error] Output already exists: {out_path} (use --force to overwrite)", file=sys.stderr)
        sys.exit(1)

    files = collect_txt_files(in_dir, args.recursive)
    if not files:
        print(f"[warn] No .txt files found in {in_dir}", file=sys.stderr)
        sys.exit(0)

    files = [p for p in files if p.resolve() != out_path]

    if not args.quiet:
        print(f"[info] Found {len(files)} .txt files under: {in_dir}")
        print(f"[info] Writing corpus to: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    with open(out_path, "wb") as outf:
        for idx, fp in enumerate(files, start=1):
            if not args.quiet:
                print(f"  [{idx:>4}/{len(files)}] {fp}")
            with open(fp, "rb") as inf:
                copied = shutil.copyfileobj(inf, outf, length=1024 * 1024)  
            try:
                total_bytes += fp.stat().st_size
            except OSError:
                pass
            if idx != len(files):
                outf.write(b"\n")

    if not args.quiet:
        size_mb = total_bytes / (1024 * 1024)
        print(f"[done] Concatenated {len(files)} files, ~{size_mb:.2f} MB")

if __name__ == "__main__":
    main()
