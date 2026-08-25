#!/usr/bin/env python3
"""引き継ぎデータの完全性を確認し、基準の数字が再現できるかを検証する。

GMO の実データに触る**前**に走らせること。ここで基準が再現しないなら、
データかコードのどちらかが壊れているので、先へ進んでも意味がない。

Example:
    python verify_bundle.py --repo /path/to/freespace
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

# 基準値（このバンドルを作った環境で実測。許容差は Sharpe ±0.01）
EXPECTED = {"全期間": 1.577, "開発期間": 1.696, "ホールドアウト": 1.059}
TOL = 0.01


def check_sums(root: Path) -> bool:
    sums = root / "SHA256SUMS"
    if not sums.exists():
        print("✗ SHA256SUMS がありません")
        return False
    bad = []
    for line in sums.read_text().splitlines():
        want, rel = line.split("  ", 1)
        path = root / rel
        if not path.exists():
            bad.append(f"欠落: {rel}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != want:
            bad.append(f"不一致: {rel}")
    if bad:
        print("✗ ファイルの検証に失敗しました:")
        for b in bad:
            print(f"    {b}")
        return False
    print(f"○ ファイル検証 OK（{len(sums.read_text().splitlines())} 件）")
    return True


def check_baseline(root: Path, repo: Path) -> bool:
    """同梱の価格データで gmo_validate.py を走らせ、基準の Sharpe が出るか確かめる。"""
    script = repo / "scripts" / "gmo_validate.py"
    if not script.exists():
        print(f"✗ {script} が見つかりません（--repo でリポジトリの場所を指定してください）")
        return False
    cmd = [sys.executable, str(script), "--dir", str(root / "prices" / "perp_1h"),
           "--config", str(repo / "configs" / "gmo_live.yaml"),
           "--symbols", "BTC", "ETH", "XRP", "BNB", "DOGE",
           "--out", str(root / "_check")]
    print(f"\n実行: {' '.join(cmd[1:])}\n")
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    print(proc.stdout or proc.stderr)
    if proc.returncode != 0:
        print("✗ 検証スクリプトが失敗しました")
        return False

    import pandas as pd
    got = pd.read_csv(root / "_check" / "summary.csv", index_col=0)["Sharpe"].to_dict()
    ok = True
    print("=== 基準との突き合わせ ===")
    for period, want in EXPECTED.items():
        have = got.get(period)
        if have is None:
            print(f"  ✗ {period}: 出力にありません")
            ok = False
            continue
        diff = abs(have - want)
        mark = "○" if diff <= TOL else "✗"
        ok &= diff <= TOL
        print(f"  {mark} {period:<10} 実測 {have:.3f} / 基準 {want:.3f}（差 {diff:.3f}）")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="..", help="freespace リポジトリのパス")
    parser.add_argument("--skip-baseline", action="store_true", help="ハッシュ確認だけ行う")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    repo = Path(args.repo).resolve()
    print(f"バンドル: {root}\nリポジトリ: {repo}\n")

    ok = check_sums(root)
    if ok and not args.skip_baseline:
        ok = check_baseline(root, repo)

    print()
    if ok:
        print("✓ 検証を通過しました。docs/HANDOFF.md §3 ステップ2（GMO 実データの取得）へ進めます。")
    else:
        print("✗ 検証に失敗しました。ここで止まってください。")
        print("  基準が再現しないまま GMO のデータを取っても、何が原因か切り分けられなくなります。")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
