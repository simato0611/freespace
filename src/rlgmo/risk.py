"""リスクレイヤ（RL の外側に置く決定論的な安全装置）。

**方針**: 学習された方策は「収益を狙う層」に限定し、資金を守る判断は
学習に任せない。RL は分布外の入力に対して何をするか保証がないため、
最終的なポジションは必ずこのレイヤを通してから発注する。

チェック項目:
    - 日次損失上限（到達したら当日フラット・新規停止）
    - 最大ドローダウンによる縮小・停止
    - ボラティリティ上限（急変時はサイズを絞る / 停止）
    - スプレッド異常（板が広がっているときは執行しない）
    - データ鮮度（最新足が遅延していたら発注しない）
    - 連敗ブレーキ
    - 証拠金維持率の下限（ロスカット水準からの距離）
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskLimits:
    """リスク上限。運用開始時は保守的に設定し、実績を見て緩める。"""

    max_position: float = 1.0             # 目標ポジション比率の上限（|a| ≤ 1）
    daily_loss_limit: float = 0.02        # 当日 -2% で新規停止・フラット化
    max_drawdown_stop: float = 0.10       # 累積 DD -10% で完全停止（人手で再開）
    drawdown_taper: float = 0.05          # DD -5% を超えたらサイズを線形に縮小
    max_vol_ann: float = 1.5              # 年率ボラ 150% 超は異常事態とみなす
    max_half_spread_bp: float = 8.0       # スプレッドがこれを超えたら執行しない
    max_data_staleness_sec: int = 90      # 最新足がこれ以上古ければ発注しない
    consecutive_loss_stop: int = 6        # 連敗でクールダウン
    cooldown_bars: int = 60
    min_margin_ratio: float = 1.5         # 証拠金維持率がこれを割ったら縮小（LC は 0.75）
    min_trade_delta: float = 0.1          # これ未満のポジション変更は発注しない（コスト節約）


@dataclass
class RiskState:
    """当日の実績を保持する状態。"""

    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    consecutive_losses: int = 0
    cooldown_left: int = 0
    halted: bool = False
    reasons: list[str] = field(default_factory=list)


class RiskManager:
    """目標ポジションをリスク制約でクリップする。

    Example:
        >>> rm = RiskManager(RiskLimits(), equity=1_000_000)
        >>> size, info = rm.apply(target=1.0, equity=980_000, vol_ann=0.6,
        ...                       half_spread_bp=2.0, staleness_sec=5, margin_ratio=3.0)
    """

    def __init__(self, limits: RiskLimits, equity: float) -> None:
        self.limits = limits
        self.state = RiskState(day_start_equity=equity, peak_equity=equity)

    def on_new_day(self, equity: float) -> None:
        self.state.day_start_equity = equity
        self.state.consecutive_losses = 0
        if not self.state.halted:
            self.state.reasons = []

    def on_trade_result(self, pnl: float) -> None:
        if pnl < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= self.limits.consecutive_loss_stop:
                self.state.cooldown_left = self.limits.cooldown_bars
                self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses = 0

    def apply(
        self,
        target: float,
        equity: float,
        vol_ann: float,
        half_spread_bp: float,
        staleness_sec: float,
        margin_ratio: float,
        current: float = 0.0,
    ) -> tuple[float, dict]:
        """目標ポジションに全制約を適用し、発注してよいサイズを返す。

        Args:
            target: 方策が出した目標ポジション比率（-1〜1）。
            equity: 現在の有効証拠金。
            vol_ann: 直近の年率ボラティリティ推定。
            half_spread_bp: 現在の片道スプレッド (bp)。
            staleness_sec: 最新足の遅延（秒）。
            margin_ratio: 現在の証拠金維持率。
            current: 現在のポジション比率。

        Returns:
            (発注すべき目標ポジション, 判定内容の dict)
        """
        lim, st = self.limits, self.state
        st.peak_equity = max(st.peak_equity, equity)
        reasons: list[str] = []
        size = max(-lim.max_position, min(lim.max_position, target))

        drawdown = 1 - equity / st.peak_equity if st.peak_equity > 0 else 0.0
        day_pnl = equity / st.day_start_equity - 1 if st.day_start_equity > 0 else 0.0

        if drawdown >= lim.max_drawdown_stop:
            st.halted = True
            reasons.append(f"max_drawdown_stop({drawdown:.1%})")
            st.reasons = list(reasons)  # 停止理由を保持（再開は人手で行う）
        if st.halted:
            return 0.0, {"size": 0.0, "halted": True, "reasons": reasons or st.reasons}

        if day_pnl <= -lim.daily_loss_limit:  # 当日は撤退
            return 0.0, {"size": 0.0, "halted": False, "reasons": ["daily_loss_limit"]}
        if st.cooldown_left > 0:
            st.cooldown_left -= 1
            return 0.0, {"size": 0.0, "halted": False, "reasons": ["cooldown"]}
        if staleness_sec > lim.max_data_staleness_sec:  # データが来ていない = 相場が見えていない
            return current, {"size": current, "halted": False, "reasons": ["stale_data:hold"]}
        if half_spread_bp > lim.max_half_spread_bp:    # 板が壊れている
            return current, {"size": current, "halted": False, "reasons": ["wide_spread:hold"]}

        if drawdown > lim.drawdown_taper:  # DD に応じて線形に縮小
            taper = max(0.0, 1 - (drawdown - lim.drawdown_taper) / (lim.max_drawdown_stop - lim.drawdown_taper))
            size *= taper
            reasons.append(f"dd_taper({taper:.2f})")
        if vol_ann > lim.max_vol_ann:      # 異常ボラでは縮小
            size *= lim.max_vol_ann / vol_ann
            reasons.append("vol_cap")
        if margin_ratio < lim.min_margin_ratio:  # ロスカットから遠ざける
            size = min(abs(size), abs(current) * 0.5) * (1 if size >= 0 else -1)
            reasons.append("margin_taper")
        if abs(size - current) < lim.min_trade_delta:  # 小さすぎる調整は見送る
            size = current
            reasons.append("below_min_delta")

        return float(size), {"size": float(size), "halted": False, "reasons": reasons,
                             "drawdown": drawdown, "day_pnl": day_pnl}
