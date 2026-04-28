from enum import Enum
from typing import Optional, Dict, List
from collections import deque, Counter
import math
import time
import os

# ---------------------------
# Exchange API stubs (platform will provide actual implementations)
# ---------------------------
class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    TEAM_A = 0  # Home-team win contract

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Platform to implement actual market order."""
    pass

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Platform to implement limit/IOC order. Return order_id if applicable."""
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Platform to implement cancel. Return success boolean."""
    return False

# ---------------------------
# Improved Strategy Implementation
# ---------------------------
class Strategy:
    """
    Improved in-play trading strategy for basketball market (TEAM_A):
      - Hybrid logistic win-probability model (score diff, nonlinear time, possession, lineup)
      - New alpha features: scoring-run detector, shot-distance clustering, foul imbalance
      - Regime awareness: close game / blowout / late-game winddown / garbage-time filter
      - Volatility-aware, conservative Kelly-style sizing with hard caps
      - Smart execution: IOC-limit at best price with market fallback, per-fill risk checks
      - Robust logging for offline analysis
    """

    def reset_state(self) -> None:
        # Core game state
        self.home_score = 0
        self.away_score = 0
        self.position = 0.0            # net contracts (positive = long HOME)
        self.capital = 100000.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

        # Market snapshot
        self.best_bid: Optional[float] = None
        self.best_ask: Optional[float] = None
        self.total_time: Optional[float] = None
        self.offset: Optional[float] = None
        self.offset_set = False

        # Model weights (conservative defaults)
        self.a = 0.32   # score differential
        self.b = 0.70   # nonlinear time weight amplitude
        self.c = 0.06   # possession
        self.d = 0.03   # lineup strength

        # lineup / players
        self.player_weights: Dict[str, float] = {f"Player {i}": 1.0 for i in range(1,51)}
        self.home_on_court: List[str] = []
        self.away_on_court: List[str] = []

        # Event history for momentum/scoring runs/shot clusters
        self.recent_events = deque(maxlen=60)   # recent event types for run detection
        self.recent_mid = deque(maxlen=40)      # history of normalized mid prices for vol
        self.recent_shot_distances = deque(maxlen=80)  # shot distances to cluster
        self.foul_count = {"home": 0, "away": 0}
        self.score_run_counter = deque(maxlen=30)  # track point swings (home - away) per short window

        # Risk & execution knobs
        self.min_edge_base = 0.012    # base minimum edge (1.2%)
        self.max_fraction = 0.20      # max capital fraction per trade
        self.max_total_exposure = 0.40
        self.stop_loss_frac = 0.06    # flatten on >6% unrealized loss
        self.take_profit_frac = 0.12  # partial take on >12% unrealized gain
        self.min_secs_between_trades = 1.0
        self.last_trade_game_time: Optional[float] = None
        self.cooldown_until_game_time: Optional[float] = None

        # Regime thresholds
        self.blowout_score_margin = 20    # detect blowouts
        self.close_game_threshold_secs = 120.0  # final 2 minutes defined as close-game sensitive
        self.late_game_winddown_secs = 60.0

        # Volatility helpers
        self.vol_floor = 1e-4

        # Logging
        self.log_path = "improved_trades.log"
        if not os.path.exists(self.log_path):
            with open(self.log_path, "a") as f:
                f.write("wall_time,game_time,event,p_model,p_market,edge,side,qty,price,position,unrealized,realized,notes\n")

    def __init__(self) -> None:
        self.reset_state()

    # ---------------------------
    # Helpers: market / mid / volatility / exposure
    # ---------------------------
    def _mid_price(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return 0.5 * (self.best_bid + self.best_ask)

    def _mid_normalized(self) -> Optional[float]:
        mid = self._mid_price()
        if mid is None or mid <= 0.0:
            return None
        return mid / 100.0

    def _append_mid(self) -> None:
        mid = self._mid_normalized()
        if mid is not None:
            self.recent_mid.append(mid)

    def _volatility(self) -> float:
        h = list(self.recent_mid)
        n = len(h)
        if n < 2:
            return self.vol_floor
        mean = sum(h) / n
        var = sum((x - mean) ** 2 for x in h) / (n - 1)
        return max(math.sqrt(var), self.vol_floor)

    def _exposure_fraction(self) -> float:
        mid = self._mid_normalized()
        if mid is None or self.capital <= 0:
            return 0.0
        exposure_value = abs(self.position) * mid * 100.0
        return exposure_value / max(1e-9, self.capital)

    def _log(self, game_time, event, p_model, p_market, edge, side, qty, price, notes=""):
        line = ",".join(map(str, [
            time.time(),
            game_time if game_time is not None else "",
            event,
            round(p_model,6),
            round(p_market,6),
            round(edge,6),
            side if isinstance(side, str) else (side.name if side is not None else ""),
            round(qty,6),
            round(price,6),
            round(self.position,6),
            round(self.unrealized_pnl,6),
            round(self.realized_pnl,6),
            notes
        ])) + "\n"
        with open(self.log_path, "a") as f:
            f.write(line)

    # Execution wrapper: prefer IOC limit at best price, fallback to market
    def _execute_order(self, side: Side, qty: float) -> None:
        if qty <= 0:
            return
        price_for_log = 0.0
        try:
            if side == Side.BUY:
                if self.best_ask is not None and self.best_ask > 0:
                    price_for_log = self.best_ask
                    order_id = place_limit_order(Side.BUY, Ticker.TEAM_A, qty, price=self.best_ask, ioc=True)
                    if not order_id:
                        place_market_order(Side.BUY, Ticker.TEAM_A, qty)
                else:
                    place_market_order(Side.BUY, Ticker.TEAM_A, qty)
            else:
                if self.best_bid is not None and self.best_bid > 0:
                    price_for_log = self.best_bid
                    order_id = place_limit_order(Side.SELL, Ticker.TEAM_A, qty, price=self.best_bid, ioc=True)
                    if not order_id:
                        place_market_order(Side.SELL, Ticker.TEAM_A, qty)
                else:
                    place_market_order(Side.SELL, Ticker.TEAM_A, qty)
        except Exception:
            # fallback to market order if limit fails for any reason
            if side == Side.BUY:
                place_market_order(Side.BUY, Ticker.TEAM_A, qty)
            else:
                place_market_order(Side.SELL, Ticker.TEAM_A, qty)

        # approximate immediate log (fills will correct realized/unrealized when on_account_update fires)
        self._log(self.last_trade_game_time, "EXECUTE", 0.0, (self._mid_price() or 0.0)/100.0, 0.0, side, qty, price_for_log, notes="execute_attempt")

    # ---------------------------
    # Callbacks required by platform
    # ---------------------------
    def on_trade_update(self, ticker: Ticker, side: Side, price: float, quantity: float) -> None:
        # Not used for model logic but could be used to enrich microstructure
        return

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        if ticker != Ticker.TEAM_A:
            return
        # maintain simple best bid/ask view
        if side == Side.BUY:
            if quantity > 0:
                if self.best_bid is None or price > self.best_bid:
                    self.best_bid = price
            else:
                if self.best_bid == price:
                    self.best_bid = None
        else:
            if quantity > 0:
                if self.best_ask is None or price < self.best_ask:
                    self.best_ask = price
            else:
                if self.best_ask == price:
                    self.best_ask = None
        self._append_mid()

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        if ticker != Ticker.TEAM_A:
            return
        if bids:
            self.best_bid = bids[0][0]
        if asks:
            self.best_ask = asks[0][0]
        self._append_mid()

    def on_account_update(self, ticker: Ticker, side: Side, price: float, quantity: float, capital_remaining: float) -> None:
        if ticker != Ticker.TEAM_A:
            return

        prev_pos = self.position
        self.capital = capital_remaining

        # Handle fills: update position, avg entry, realized pnl robustly
        if side == Side.BUY:
            new_pos = self.position + quantity
            if prev_pos >= 0:
                prev_value = prev_pos * self.avg_entry_price
                buy_value = quantity * price
                total_qty = prev_pos + quantity
                self.avg_entry_price = (prev_value + buy_value) / total_qty if total_qty > 0 else price
            else:
                # reducing short -> realize pnl
                realized = quantity * (self.avg_entry_price - price)
                self.realized_pnl += realized
                # if flipping to long, set new avg_entry to price for flipped part
                if quantity > abs(prev_pos):
                    flipped = quantity - abs(prev_pos)
                    self.avg_entry_price = price
            self.position = new_pos
        else:  # SELL
            new_pos = self.position - quantity
            if prev_pos <= 0:
                prev_value = abs(prev_pos) * self.avg_entry_price
                sell_value = quantity * price
                total_qty = abs(prev_pos) + quantity
                self.avg_entry_price = (prev_value + sell_value) / total_qty if total_qty > 0 else price
            else:
                # reducing long -> realize pnl
                realized = quantity * (price - self.avg_entry_price)
                self.realized_pnl += realized
                if quantity > prev_pos:
                    flipped = quantity - prev_pos
                    self.avg_entry_price = price
            self.position = new_pos

        # update unrealized using mid
        mid = self._mid_price()
        if mid is not None:
            self.unrealized_pnl = self.position * (mid - self.avg_entry_price)
        else:
            self.unrealized_pnl = 0.0

        # immediate post-fill risk actions (flatten on stoploss / take profit)
        self._post_fill_risk_checks()

    def _post_fill_risk_checks(self) -> None:
        # stop-loss: flatten if unrealized < -stop_loss_frac * capital
        if self.capital <= 0:
            return
        if self.unrealized_pnl < -self.stop_loss_frac * self.capital:
            # flatten
            if self.position > 0:
                self._execute_order(Side.SELL, abs(self.position))
            elif self.position < 0:
                self._execute_order(Side.BUY, abs(self.position))
            # set short cooldown: skip trades for a short window
            self.cooldown_until_game_time = (self.last_trade_game_time - 0.0) if self.last_trade_game_time is not None else None
            return

        # take-profit: partially reduce exposure
        if self.unrealized_pnl > self.take_profit_frac * self.capital:
            qty = 0.5 * abs(self.position)
            if qty > 0:
                if self.position > 0:
                    self._execute_order(Side.SELL, qty)
                else:
                    self._execute_order(Side.BUY, qty)
                self.cooldown_until_game_time = (self.last_trade_game_time - 0.0) if self.last_trade_game_time is not None else None

    # ---------------------------
    # Feature engineering: runs, shot-distance clusters, foul imbalance
    # ---------------------------
    def _update_event_history(self, event_type, home_away, shot_type, coordinate_x, coordinate_y, partial_points=0):
        # push event types for run detection and shot clustering
        ev = {"etype": event_type, "who": home_away, "pts": partial_points}
        self.recent_events.append(ev)
        # scoring run: track last N point differential changes
        if event_type == "SCORE" and partial_points is not None:
            self.score_run_counter.append((home_away, partial_points))
        # shot distance
        if event_type in ("SCORE", "MISSED") and coordinate_x is not None and coordinate_y is not None:
            # approximate distance from hoop at (25, 5) - court coordinate systems vary; this is heuristic
            dx = coordinate_x - 25.0
            dy = coordinate_y - 5.0
            dist = math.hypot(dx, dy)
            self.recent_shot_distances.append(dist)
        # fouls
        if event_type == "FOUL":
            if home_away in ("home", "away"):
                self.foul_count[home_away] = self.foul_count.get(home_away, 0) + 1

    def _scoring_run_strength(self) -> float:
        # compute net points for home over last window (recent score events)
        window = 12
        h = list(self.score_run_counter)[-window:]
        net = 0
        for who, pts in h:
            if who == "home":
                net += pts
            else:
                net -= pts
        # normalize roughly to [-1,1] by dividing by 12*3 (assume typical 3-pt bursts)
        return max(-1.0, min(1.0, net / (window * 3.0)))

    def _shot_cluster_indicator(self) -> float:
        # Returns a small factor indicating whether recent shots are long-range heavy (favoring home if they are hitting 3s)
        if not self.recent_shot_distances:
            return 0.0
        avg = sum(self.recent_shot_distances) / len(self.recent_shot_distances)
        # if average distance > threshold, indicate more 3pt attempts (positive), else negative
        return (avg - 20.0) / 30.0  # scaled roughly to [-1,1] for typical distances

    def _foul_imbalance(self) -> float:
        # positive if home has more fouls (bad), negative if away has more fouls (good for home)
        h = self.foul_count.get("home", 0)
        a = self.foul_count.get("away", 0)
        diff = h - a
        # scale: each foul ~ small effect
        return max(-1.0, min(1.0, -0.02 * diff))  # negative diff benefits home

    # ---------------------------
    # Main game-event handler
    # ---------------------------
    def on_game_event_update(
        self,
        event_type: str,
        home_away: str,
        home_score: int,
        away_score: int,
        player_name: Optional[str],
        substituted_player_name: Optional[str],
        shot_type: Optional[str],
        assist_player: Optional[str],
        rebound_type: Optional[str],
        coordinate_x: Optional[float],
        coordinate_y: Optional[float],
        time_seconds: Optional[float]
    ) -> None:
        # Update scores & basic state
        prev_home = self.home_score
        prev_away = self.away_score
        self.home_score = home_score
        self.away_score = away_score

        # update possession heuristic on events that imply possession change
        if event_type in ("SCORE", "MISSED", "TURNOVER"):
            self.possession = "away" if home_away == "home" else "home"

        # substitutions & initial on-court setup
        if event_type == "SUBSTITUTION" and player_name and substituted_player_name:
            if home_away == "home":
                if substituted_player_name in self.home_on_court:
                    self.home_on_court.remove(substituted_player_name)
                self.home_on_court.append(player_name)
            else:
                if substituted_player_name in self.away_on_court:
                    self.away_on_court.remove(substituted_player_name)
                self.away_on_court.append(player_name)
        elif event_type in ("JUMP_BALL", "START_PERIOD"):
            if not self.home_on_court:
                self.home_on_court = [f"Player {i}" for i in range(1,6)]
            if not self.away_on_court:
                self.away_on_court = [f"Player {i}" for i in range(6,11)]

        # set total_time if available (first START_PERIOD/JUMP_BALL)
        if self.total_time is None and event_type in ("START_PERIOD", "JUMP_BALL") and time_seconds:
            self.total_time = time_seconds
        if self.total_time is None:
            # fallback if not provided (midgame join)
            self.total_time = 2880.0

        # early exit: stop trading in last X seconds (tight winddown) OR if we explicitly want to stop trading
        if time_seconds is not None and time_seconds <= 0.0:
            # events may still occur near 0, but treat END_GAME separately
            pass

        # update event history features
        pts_scored = None
        if event_type == "SCORE":
            # infer points from score delta
            delta_home = self.home_score - prev_home
            delta_away = self.away_score - prev_away
            if home_away == "home":
                pts_scored = delta_home
            else:
                pts_scored = delta_away
        self._update_event_history(event_type, home_away, shot_type, coordinate_x, coordinate_y, partial_points=pts_scored if pts_scored is not None else 0)

        # append mid and compute volatility
        self._append_mid()
        vol = self._volatility()

        # Stop trading late or if in blowout
        score_diff = self.home_score - self.away_score
        abs_diff = abs(score_diff)

        # Detect blowout: reduce or stop trading if margin is huge relative to remaining time
        blowout_active = False
        if abs_diff >= self.blowout_score_margin:
            # if margin high and there is substantial time left, label blowout
            blowout_active = True

        # If blowout or very late garbage time, throttle aggressively
        if blowout_active and time_seconds is not None and time_seconds > self.late_game_winddown_secs:
            # reduce trading aggressively
            return

        # Set t_frac and nonlinear time weight (amplify late game)
        t_frac = time_seconds / self.total_time if (time_seconds is not None and self.total_time) else 0.0
        time_weight_nl = (1.0 - t_frac) ** 2

        # Lineup strength
        home_strength = sum(self.player_weights.get(p,1.0) for p in self.home_on_court) if self.home_on_court else 0.0
        away_strength = sum(self.player_weights.get(p,1.0) for p in self.away_on_court) if self.away_on_court else 0.0
        lineup_strength = home_strength - away_strength

        # possession factor (scaled by late-game sensitivity)
        possession_factor = 0.0
        if hasattr(self, "possession"):
            if getattr(self, "possession", None) == "home":
                possession_factor = 1.0 * time_weight_nl
            elif getattr(self, "possession", None) == "away":
                possession_factor = -1.0 * time_weight_nl

        # Extra alpha features
        run_strength = self._scoring_run_strength()            # [-1,1]
        shot_cluster = self._shot_cluster_indicator()          # approx [-1,1]
        foul_imb = self._foul_imbalance()                      # [-1,1]

        # Calibrate offset on first meaningful mid observation
        mid_norm = self._mid_normalized()
        if mid_norm is not None:
            if not self.offset_set:
                p_market = min(max(mid_norm, 0.01), 0.99)
                logit = math.log(p_market / (1.0 - p_market))
                model_logit = (self.a * score_diff + self.b * time_weight_nl + self.c * possession_factor + self.d * lineup_strength)
                # incorporate small run & shot signals into initial calibration (lightly)
                augment = 0.18 * run_strength + 0.06 * shot_cluster + 0.04 * foul_imb
                self.offset = logit - (model_logit + augment)
                self.offset_set = True
            else:
                # gentle recalibration if market drifts significantly (robust learning)
                if len(self.recent_mid) >= max(6, self.recent_mid.maxlen // 2):
                    recent_mean = sum(self.recent_mid) / len(self.recent_mid)
                    if abs(mid_norm - recent_mean) > 0.07 and self.offset is not None:
                        p_market = min(max(mid_norm, 0.01), 0.99)
                        new_logit = math.log(p_market / (1.0 - p_market))
                        model_logit = (self.a * score_diff + self.b * time_weight_nl + self.c * possession_factor + self.d * lineup_strength)
                        target_offset = new_logit - model_logit
                        alpha = 0.08
                        self.offset = (1.0 - alpha) * self.offset + alpha * target_offset

        # compute model probability (numerically stable logistic)
        x = (self.a * score_diff +
             self.b * time_weight_nl +
             self.c * possession_factor +
             self.d * lineup_strength +
             (self.offset if self.offset is not None else 0.0) +
             0.16 * run_strength +       # incorporate run strength
             0.04 * shot_cluster +       # small influence of shot-cluster
             0.03 * foul_imb)            # small influence of foul imbalance

        # numerically stable logistic
        if x >= 0:
            p_home_win = 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            p_home_win = exp_x / (1.0 + exp_x)

        # If we don't have market info, don't trade
        mid_price = self._mid_price()
        if mid_price is None:
            return
        p_market = mid_price / 100.0
        edge = p_home_win - p_market

        # adaptive min edge: base + volatility term
        dynamic_min_edge = self.min_edge_base + min(0.06, vol * 1.8)
        # relax threshold slightly if very late and game close (allow decisive bets), else tighten
        if time_seconds is not None and time_seconds <= self.close_game_threshold_secs and abs_diff <= 5:
            dynamic_min_edge *= 0.85

        # throttle with last trade time
        if self.last_trade_game_time is not None and time_seconds is not None:
            elapsed = abs(self.last_trade_game_time - time_seconds)
            if elapsed < self.min_secs_between_trades:
                return

        # exposure cap
        if self._exposure_fraction() >= self.max_total_exposure:
            return

        # require significant edge
        if abs(edge) <= dynamic_min_edge:
            return

        # regime adjustments: reduce sizing in blowout or garbage time
        if blowout_active:
            # ignore small edges in blowouts
            if abs(edge) < 0.06:
                return

        # compute sizing using squared-edge scaled by inverse vol (Kelly-style intuition but conservative)
        k = 1.0
        raw_frac = k * (edge * edge) / max(vol, self.vol_floor)

        # reduce bet size late in game (winddown) to avoid end-game randomness
        if time_seconds is not None and time_seconds <= self.late_game_winddown_secs:
            raw_frac *= 0.4

        # cap by max_fraction and remaining exposure allowance
        frac = min(self.max_fraction, raw_frac)
        remaining_allowance = max(0.0, self.max_total_exposure - self._exposure_fraction())
        frac = min(frac, remaining_allowance)
        if frac <= 0:
            return

        # compute qty depending on side using best_ask/bid
        if edge > 0:
            # Buy home (long)
            if self.best_ask is None or self.best_ask <= 0:
                return
            qty = (frac * self.capital) / self.best_ask
            if qty < 1e-6:
                return
            # cooldown_until_game_time check
            if self.cooldown_until_game_time is not None and time_seconds is not None and time_seconds <= self.cooldown_until_game_time:
                return
            # execute IOC-limit at ask (preferred)
            self._execute_order(Side.BUY, qty)
            self.last_trade_game_time = time_seconds
            self._log(time_seconds, event_type, p_home_win, p_market, edge, "BUY", qty, self.best_ask or 0.0,
                      notes=f"edge_buy run{run_strength:.3f} shot{shot_cluster:.3f} foul{foul_imb:.3f}")
        else:
            # Sell home (go short)
            if self.best_bid is None or self.best_bid <= 0:
                return
            qty = (frac * self.capital) / self.best_bid
            if qty < 1e-6:
                return
            if self.cooldown_until_game_time is not None and time_seconds is not None and time_seconds <= self.cooldown_until_game_time:
                return
            self._execute_order(Side.SELL, qty)
            self.last_trade_game_time = time_seconds
            self._log(time_seconds, event_type, p_home_win, p_market, edge, "SELL", qty, self.best_bid or 0.0,
                      notes=f"edge_sell run{run_strength:.3f} shot{shot_cluster:.3f} foul{foul_imb:.3f}")

        # Reset state at end of game
        if event_type == "END_GAME":
            self.reset_state()

    # Optional cleanup if platform provides separate end hook
    def on_end_game_cleanup(self) -> None:
        self.reset_state()
