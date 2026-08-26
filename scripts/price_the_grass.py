#!/usr/bin/env python3
"""他人の「実績」に値段をつける。

SNS や業者資料で見る収益実績は、たいてい**都合の良い部分だけ**が書いてある。
年利、勝率、最大ドローダウン——並んでいる数字そのものは嘘でなくても、
**それらの組み合わせが物理的にあり得るか**は誰も検算しない。

このスクリプトは、公開されている数字だけから次を検算する。

1. **勝率と Sharpe の整合性** — 正規分布ならこの Sharpe でこの勝率になるはず、を計算する。
   実際の勝率がそれより高いなら、**負に歪んでいる**（たまに大きく負ける形）
2. **最大DD の妥当性** — その Sharpe・その期間なら DD はこれくらい出るはず、を
   モンテカルロで出す。申告 DD が小さすぎるなら、期間が短いか運が良かっただけ
3. **運と実力を見分けるのに必要な期間** — その Sharpe が本物だと言うには何年要るか
4. **試行回数による割引** — N 通り試した中の最良なら、その Sharpe はどこまで割引くべきか

**このスクリプトは「儲かるか」を判定しない。「主張が数字として成立するか」を検算する。**

Example:
    # SNS でよく見る形: 年利20%、月次勝率95%、最大DD 0.8%、運用1年
    python scripts/price_the_grass.py --annual 0.20 --vol 0.06 --win-rate 0.95 --months 12

    # 自分の戦略を同じ物差しに乗せる
    python scripts/price_the_grass.py --annual 0.19 --vol 0.164 --win-rate 0.60 --months 15
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.stats import binomtest, norm


def implied_win_rate(sharpe_ann: float, periods_per_year: int) -> float:
    """正規分布を仮定したときの、その Sharpe に対応する期間勝率。"""
    return float(norm.cdf(sharpe_ann / math.sqrt(periods_per_year)))


def expected_max_dd(sharpe_ann: float, vol_ann: float, months: int, n: int = 20000) -> dict:
    """その Sharpe・その期間で、最大DD がどのくらい出るはずかをモンテカルロで出す。"""
    rng = np.random.default_rng(0)
    mu_m = sharpe_ann * vol_ann / 12
    sd_m = vol_ann / math.sqrt(12)
    paths = rng.normal(mu_m, sd_m, size=(n, months))
    eq = np.cumprod(1 + paths, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd = (eq / peak - 1).min(axis=1)
    return {"中央値": float(np.median(dd)), "10%点": float(np.percentile(dd, 10)),
            "90%点": float(np.percentile(dd, 90))}


def years_to_prove(sharpe_ann: float, t_target: float = 2.0) -> float:
    """その Sharpe が偶然でないと言うのに必要な年数（t 値が目標に届くまで）。"""
    if sharpe_ann <= 0:
        return float("inf")
    return (t_target / sharpe_ann) ** 2


def haircut_for_trials(sharpe_ann: float, n_trials: int, months: int,
                       trial_sd: float = 0.5) -> float:
    """N 通り試して最良を選んだ場合、期待される『最良の見かけ Sharpe』を差し引く。"""
    if n_trials <= 1:
        return sharpe_ann
    gamma = 0.5772156649015329
    e_max = trial_sd * ((1 - gamma) * norm.ppf(1 - 1 / n_trials)
                        + gamma * norm.ppf(1 - 1 / (n_trials * math.e)))
    return sharpe_ann - e_max


def breakeven_cost_bp(sharpe_ann: float, vol_ann: float, holding_days: float) -> float:
    """その成績を消し飛ばす片道コスト。保有期間が短いほど小さくなる。"""
    trades_per_year = 365 / max(holding_days, 1e-9)
    gross_ann = sharpe_ann * vol_ann
    return gross_ann / (2 * trades_per_year) * 1e4


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annual", type=float, required=True, help="申告の年率リターン（0.20 = 20%%）")
    p.add_argument("--vol", type=float, default=None, help="年率ボラ。分からなければ --sharpe を渡す")
    p.add_argument("--sharpe", type=float, default=None, help="申告 Sharpe（--vol の代わり）")
    p.add_argument("--max-dd", type=float, default=None, help="申告の最大DD（0.008 = 0.8%%）")
    p.add_argument("--win-rate", type=float, default=None, help="申告の月次勝率（0.95 = 95%%）")
    p.add_argument("--months", type=int, required=True, help="運用期間（か月）")
    p.add_argument("--holding-days", type=float, default=None, help="平均保有期間（日）")
    p.add_argument("--trials", type=int, default=1, help="何通り試した中の最良か")
    args = p.parse_args()

    if args.sharpe is None and args.vol is None:
        raise SystemExit("--vol か --sharpe のどちらかを渡してください")
    sharpe = args.sharpe if args.sharpe is not None else args.annual / args.vol
    vol = args.vol if args.vol is not None else args.annual / max(sharpe, 1e-9)

    print("=" * 72)
    print(f"申告値: 年率 {args.annual*100:.1f}% / 年率ボラ {vol*100:.1f}% / "
          f"Sharpe {sharpe:.2f} / 期間 {args.months} か月")
    print("=" * 72)

    flags = []

    # --- 1. 勝率と Sharpe の整合性（二項検定で判定する）
    if args.win_rate is not None:
        implied = implied_win_rate(sharpe, 12)
        gap = args.win_rate - implied
        wins = int(round(args.win_rate * args.months))
        pval = float(binomtest(wins, args.months, implied, alternative="two-sided").pvalue)
        print(f"\n【1】勝率の検算")
        print(f"  申告の月次勝率            {args.win_rate*100:.1f}%（{args.months} か月中 {wins} 勝）")
        print(f"  Sharpe {sharpe:.2f} が示す勝率   {implied*100:.1f}%（正規分布を仮定）")
        print(f"  差                       {gap*100:+.1f}ポイント   二項検定 p = {pval:.4f}")
        if pval < 0.05 and gap > 0:
            print(f"  → **負に歪んでいる（有意）。** 勝率が高すぎる。")
            print(f"     小さく勝ち続け、たまに大きく負ける形。")
            print(f"     キャリー取引・オプション売り・ナンピンがこの形になる。")
            print(f"     この形は「まだ大きく負けていないだけ」の可能性が常にある。")
            flags.append(f"勝率が Sharpe に対して高すぎる（負の歪度, p={pval:.3f}）")
        elif pval < 0.05 and gap < 0:
            print(f"  → 正に歪んでいる（有意）。勝率は低いが勝つときに大きい。トレンドフォロー型。")
        else:
            # 検出力: この期間なら何ポイントの差まで見えるか
            se = math.sqrt(implied * (1 - implied) / args.months)
            detectable = 1.96 * se
            print(f"  → 有意差なし。ただし **この期間で検出できる差は ±{detectable*100:.1f} ポイント**。")
            if abs(gap) > detectable * 0.5:
                print(f"     差 {gap*100:+.1f} はその範囲に埋もれている。**判定できないだけで、")
                print(f"     歪んでいないという意味ではない。**")
                need_m = int(math.ceil((1.96 / max(abs(gap), 1e-9)) ** 2 * implied * (1 - implied)))
                print(f"     この差を有意に検出するには **{need_m} か月**（{need_m/12:.1f} 年）必要。")
                flags.append(f"期間 {args.months} か月では歪みを判定できない"
                             f"（判定に {need_m} か月必要）")
            else:
                print(f"     差 {gap*100:+.1f} は小さく、極端な歪みは無さそう。")

    # --- 2. 最大DD の妥当性
    if args.max_dd is not None:
        exp = expected_max_dd(sharpe, vol, args.months)
        print(f"\n【2】最大ドローダウンの検算")
        print(f"  申告の最大DD              {args.max_dd*100:.1f}%")
        print(f"  この Sharpe・この期間で出るはずの DD:")
        print(f"      中央値 {exp['中央値']*100:.1f}%   よく出る範囲 {exp['90%点']*100:.1f}% 〜 {exp['10%点']*100:.1f}%")
        if args.max_dd > -exp["中央値"] * 0.4:
            pass
        if -args.max_dd > exp["90%点"]:
            print(f"  → **申告 DD が小さすぎる。** 考えられるのは 3 つ:")
            print(f"     (a) 期間が短くてまだ悪い局面が来ていない")
            print(f"     (b) リターンが正規分布でない（左の裾が未実現）")
            print(f"     (c) 申告が正確でない")
            flags.append("最大DD が Sharpe と期間に対して小さすぎる")
        else:
            print(f"  → 整合的な範囲。")

    # --- 3. 運と実力の識別に要する期間
    need = years_to_prove(sharpe)
    print(f"\n【3】その成績が偶然でないと言うのに必要な期間")
    print(f"  必要年数（t値 2.0 に届くまで）  {need:.1f} 年")
    print(f"  実際の期間                    {args.months/12:.1f} 年")
    if args.months / 12 < need:
        print(f"  → **期間が足りない。** この実績では、真の Sharpe がゼロである可能性を排除できない。")
        flags.append(f"統計的な確証に {need:.1f} 年必要だが実績は {args.months/12:.1f} 年")
    else:
        print(f"  → 期間は足りている。")

    # --- 4. 試行回数による割引
    if args.trials > 1:
        adj = haircut_for_trials(sharpe, args.trials, args.months)
        print(f"\n【4】試行回数による割引")
        print(f"  {args.trials} 通り試した中の最良なら、偶然だけで出る Sharpe を差し引く必要がある")
        print(f"  申告 Sharpe {sharpe:.2f} → 割引後 {adj:.2f}")
        if adj <= 0:
            print(f"  → **割引後にゼロ以下。** その成績は探索の副産物である可能性が高い。")
            flags.append("試行回数を考慮すると Sharpe がゼロ以下になる")

    # --- 5. コスト耐性
    if args.holding_days is not None:
        be = breakeven_cost_bp(sharpe, vol, args.holding_days)
        print(f"\n【5】コスト耐性")
        print(f"  平均保有 {args.holding_days:.1f} 日 → 年間 {365/args.holding_days:.0f} 回転")
        print(f"  この成績を消す片道コスト        {be:.1f}bp")
        if be < 5:
            print(f"  → **余裕が無い。** 実効スプレッドが少しずれるだけで消える。")
            flags.append(f"損益分岐コストが片道 {be:.1f}bp しかない")
        else:
            print(f"  → 余裕がある（実勢 1〜10bp に対して {be:.1f}bp）。")

    # --- まとめ
    print("\n" + "=" * 72)
    if flags:
        print(f"検出された疑問点 {len(flags)} 件:")
        for f in flags:
            print(f"  ✗ {f}")
    else:
        print("数字の整合性に、明らかな矛盾は見つからなかった。")
    print("=" * 72)
    print("""
数字で検算できないこと（必ず別途確認する）:
  ・危機局面を経験したか（大暴落を通過していない実績は、通過していないというだけ）
  ・キャパシティ（100万円で動くものが1億円で動くとは限らない）
  ・生存者バイアス（負けた人は投稿しない）
  ・執行の前提（板の厚み、約定率、手数料階層を持っているか）
  ・カウンターパーティ（取引所が飛んだら全部ゼロ）
""")


if __name__ == "__main__":
    main()
