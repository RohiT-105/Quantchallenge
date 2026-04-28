from enum import Enum
from typing import Optional, Dict, List
from collections import deque
import math
import time
import os

# ---------------------------
# Exchange API stubs (platform provides these on submit)
# ---------------------------
class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    TEAM_A = 0  # home-team win contract

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Platform should implement actual market order."""
    pass

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Platform should implement limit/IOC order and return order_id (or 0/None)."""
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Platform should implement cancel; return True if canceled."""
    return False

# ---------------------------
# Hybrid Event + Indicator Strategy
# ---------------------------
class Strategy:
    """
    Hybrid in-play strategy combining market indicators and game events.
    - Uses mid-price momentum & volatility (from orderbook) for timing.
    - Uses game events (SCORE, MISSED, TURNOVER, FOUL, SUBSTITUTION, POSSESSION heuristics)
      to compute a fast in-game win probability model.
    - Conservative risk management: per-trade and total exposure caps, stop-loss, take-profit,
      cooldowns, late-game winddown.
    """

    def reset_state(self) -> None:
        # ---------- Market & Position State ----------
        self.best_bid: Optional[float] = None   # price in range [0..100]
        self.best_ask: Optional[float] = None
        self.position: float = 0.0              # net contracts (positive = long HOME)
        self.capital: float = 100000.0          # bank
        self.avg_entry_price: float = 0.0
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0

        # ---------- Game State ----------
        self.home_score: int = 0
        self.away_score: int = 0
        self.possession: Optional[str] = None   # "home", "away", or None
        self.total_time: Optional[float] = None
        self.time_seconds: Optional[float] = None

        # lineups & player weights
        self.player_weights: Dict[str, float] = {f"Player {i}": 1.0 for i in range(1, 51)}
        self.home_on_court: List[str] = []
        self.away_on_court: List[str] = []

        # ---------- Event-derived features ----------
        self.recent_score_events = deque(maxlen=40)   # (timestamp, 'home'/'away', points)
        self.recent_shot_distances = deque(maxlen=80)
        self.foul_count = {"home": 0, "away": 0}
        self.recent_raw_events = deque(maxlen=120)

        # ---------- Market history & indicators ----------
        self.mid_history = deque(maxlen=60)  # normalized mid prices (0..1)
        self.vol_floor = 1e-4

        # ---------- Model parameters (tuneable) ----------
        self.a = 0.30  # score differential weight
        self.b = 0.62  # time weight (nonlinear)
        self.c = 0.06  # possession weight
        self.d = 0.02  # lineup strength weight
        # Small prior bias toward home at game start (logit space). Keep small.
        # Positive value slightly favors home. Set to 0.05 by default (≈ ~1-2% prob bias).
        self.starting_home_bias_logit = 0.03

        # ---------- Execution & risk parameters ----------
        self.min_edge_base = 0.014       # base min edge to consider (1.4%)
        self.max_fraction = 0.18         # max fraction of bankroll per trade
        self.max_total_exposure = 0.40   # max total exposure of bankroll
        self.stop_loss_frac = 0.06       # flatten if unrealized loss exceeds 6% of capital
        self.take_profit_frac = 0.10     # take partial profits at 10% unrealized gain
        self.min_secs_between_trades = 1.2
        self.last_trade_game_time: Optional[float] = None
        self.cooldown_until_game_time: Optional[float] = None

        # ---------- Regime thresholds ----------
        self.blowout_margin = 18          # detect blowouts >= 18 pts
        self.late_game_cutoff = 60.0      # stop new trades in last 60 seconds
        self.late_game_winddown_secs = 180.0  # reduce sizing in final 3 minutes

        # ---------- Logging ----------
        self.log_path = "hybrid_strategy.log"
        if not os.path.exists(self.log_path):
            with open(self.log_path, "a") as f:
                f.write("wall_time,game_time,event,p_model,p_market,edge,side,qty,price,position,unrealized,realized,notes\n")

    def __init__(self) -> None:
        self.reset_state()

    # --------------------
    # Utility / indicator helpers
    # --------------------
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
        m = self._mid_normalized()
        if m is not None:
            self.mid_history.append(m)

    def _volatility(self) -> float:
        h = list(self.mid_history)
        n = len(h)
        if n < 2:
            return self.vol_floor
        mean = sum(h) / n
        var = sum((x - mean) ** 2 for x in h) / (n - 1)
        return max(math.sqrt(var), self.vol_floor)

    def _momentum(self, lookback:int = 5) -> float:
        """Simple momentum as percent change over lookback mids (normalized)."""
        if len(self.mid_history) < lookback + 1:
            return 0.0
        arr = list(self.mid_history)
        prev = arr[-(lookback+1)]
        cur = arr[-1]
        if prev <= 0:
            return 0.0
        return (cur - prev) / prev

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
            round(p_model,6) if p_model is not None else "",
            round(p_market,6) if p_market is not None else "",
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

    # Execution wrapper (IOC limit then market fallback)
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
            if side == Side.BUY:
                place_market_order(Side.BUY, Ticker.TEAM_A, qty)
            else:
                place_market_order(Side.SELL, Ticker.TEAM_A, qty)

        self._log(self.time_seconds, "EXECUTE", None, (self._mid_price() or 0.0)/100.0, 0.0, side, qty, price_for_log, notes="exec_attempt")

    # --------------------
    # Orderbook / account callbacks
    # --------------------
    def on_trade_update(self, ticker: Ticker, side: Side, price: float, quantity: float) -> None:
        # not used directly in logic; kept for compatibility.
        return

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        if ticker != Ticker.TEAM_A:
            return
        # maintain best bid/ask
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

    def on_account_update(self,
                          ticker: Ticker,
                          side: Side,
                          price: float,
                          quantity: float,
                          capital_remaining: float) -> None:
        # robust position & PnL bookkeeping
        if ticker != Ticker.TEAM_A:
            return
        prev_position = self.position
        self.capital = capital_remaining

        if side == Side.BUY:
            new_position = self.position + quantity
            if prev_position >= 0:
                prev_value = prev_position * self.avg_entry_price
                new_value = quantity * price
                total_qty = prev_position + quantity
                self.avg_entry_price = (prev_value + new_value) / total_qty if total_qty > 0 else price
            else:
                # reducing short -> realize pnl
                realized = quantity * (self.avg_entry_price - price)
                self.realized_pnl += realized
                if quantity > abs(prev_position):
                    flipped_qty = quantity - abs(prev_position)
                    self.avg_entry_price = price
            self.position = new_position
        else:
            new_position = self.position - quantity
            if prev_position <= 0:
                prev_value = abs(prev_position) * self.avg_entry_price
                new_value = quantity * price
                total_qty = abs(prev_position) + quantity
                self.avg_entry_price = (prev_value + new_value) / total_qty if total_qty > 0 else price
            else:
                realized = quantity * (price - self.avg_entry_price)
                self.realized_pnl += realized
                if quantity > prev_position:
                    flipped_qty = quantity - prev_position
                    self.avg_entry_price = price
            self.position = new_position

        # update unrealized using mid
        mid = self._mid_price()
        if mid is not None:
            self.unrealized_pnl = self.position * (mid - self.avg_entry_price)
        else:
            self.unrealized_pnl = 0.0

        # immediate risk reactions
        self._post_fill_risk_checks()

    def _post_fill_risk_checks(self) -> None:
        if self.capital <= 0:
            return
        # stop-loss flatten
        if self.unrealized_pnl < -self.stop_loss_frac * self.capital:
            if self.position > 0:
                self._execute_order(Side.SELL, abs(self.position))
            elif self.position < 0:
                self._execute_order(Side.BUY, abs(self.position))
            # short cooldown
            self.cooldown_until_game_time = (self.time_seconds - 0.0) if self.time_seconds is not None else None
            return
        # take-profit: partial scale down
        if self.unrealized_pnl > self.take_profit_frac * self.capital:
            qty = 0.5 * abs(self.position)
            if qty > 0:
                if self.position > 0:
                    self._execute_order(Side.SELL, qty)
                else:
                    self._execute_order(Side.BUY, qty)
                self.cooldown_until_game_time = (self.time_seconds - 0.0) if self.time_seconds is not None else None

    # --------------------
    # Event feature engineering
    # --------------------
    def _record_score(self, who: str, pts: int) -> None:
        self.recent_score_events.append((time.time(), who, pts))

    def _scoring_run_strength(self, lookback_secs: float = 25.0) -> float:
        if not self.recent_score_events:
            return 0.0
        now = time.time()
        net = 0
        for ts, who, pts in reversed(self.recent_score_events):
            if now - ts <= lookback_secs:
                if who == "home":
                    net += pts
                else:
                    net -= pts
            else:
                break
        # normalize: assume ~12 pts is a strong short window run
        return max(-1.0, min(1.0, net / 12.0))

    def _shot_distance_trend(self) -> float:
        if not self.recent_shot_distances:
            return 0.0
        avg = sum(self.recent_shot_distances) / len(self.recent_shot_distances)
        # distances: small -> inside shots; large -> more 3pt attempts
        return (avg - 18.0) / 25.0  # roughly scale to [-1,1]

    def _foul_imbalance(self) -> float:
        h = self.foul_count.get("home", 0)
        a = self.foul_count.get("away", 0)
        diff = h - a
        # negative means away more fouls (good for home); scale small
        return max(-1.0, min(1.0, -0.03 * diff))

    # --------------------
    # Core event-driven logic
    # --------------------
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
        """
        Primary handler that reacts to game events.
        Strategy uses both event-derived signals and market indicators for trading decisions.
        """
        # update game state
        prev_home = self.home_score
        prev_away = self.away_score
        self.home_score = home_score
        self.away_score = away_score
        self.time_seconds = time_seconds

        # possession heuristic: after SCORE, TURNOVER, MISSED we infer who likely has possession
        if event_type in ("SCORE", "MISSED", "TURNOVER"):
            # event.home_away is the team associated with the event; possession likely flips
            self.possession = "away" if home_away == "home" else "home"

        # substitutions
        if event_type == "SUBSTITUTION" and player_name and substituted_player_name:
            if home_away == "home":
                if substituted_player_name in self.home_on_court:
                    self.home_on_court.remove(substituted_player_name)
                self.home_on_court.append(player_name)
            else:
                if substituted_player_name in self.away_on_court:
                    self.away_on_court.remove(substituted_player_name)
                self.away_on_court.append(player_name)

        # initial on-court setup on JUMP_BALL / START_PERIOD
        if event_type in ("JUMP_BALL", "START_PERIOD"):
            if not self.home_on_court:
                self.home_on_court = [f"Player {i}" for i in range(1,6)]
            if not self.away_on_court:
                self.away_on_court = [f"Player {i}" for i in range(6,11)]

        # track shots & distances for clustering
        if event_type in ("SCORE", "MISSED") and coordinate_x is not None and coordinate_y is not None:
            # approximate distance to rim (heuristic center)
            dx = (coordinate_x - 25.0)
            dy = (coordinate_y - 5.0)
            dist = math.hypot(dx, dy)
            self.recent_shot_distances.append(dist)

        # fouls
        if event_type == "FOUL":
            if home_away in ("home", "away"):
                self.foul_count[home_away] = self.foul_count.get(home_away, 0) + 1

        # scoring events -> record
        if event_type == "SCORE":
            # infer points by comparing prev and current score
            delta_home = self.home_score - prev_home
            delta_away = self.away_score - prev_away
            pts = delta_home if home_away == "home" else delta_away
            # weight 3-pointers/dunks slightly more if shot_type available
            weight = 1.0
            if shot_type in ("THREE_POINT", "DUNK"):
                weight = 1.3
            if shot_type == "FREE_THROW":
                weight = 0.7
            weighted_pts = max(1, int(round(pts * weight)))
            self._record_score(home_away, weighted_pts)

        # append raw event
        self.recent_raw_events.append((time.time(), event_type, home_away))

        # set total time if available
        if self.total_time is None and event_type in ("START_PERIOD", "JUMP_BALL") and time_seconds:
            self.total_time = time_seconds
        if self.total_time is None:
            self.total_time = 2880.0  # fallback

        # append market mid history & compute vol / momentum
        self._append_mid()
        vol = self._volatility()
        mom = self._momentum(lookback=6)

        # ------- trading gating rules -------
        # 1) Avoid new trades in final cutoff
        if self.time_seconds is not None and self.time_seconds <= self.late_game_cutoff:
            return

        # 2) Hard stop if bankroll has suffered severe drawdown (safety)
        if self.capital < (1.0 - 0.40) * 100000.0:  # hard stop at 40% loss
            return

        # 3) honor cooldown (used after big adverse fills or stops)
        if self.cooldown_until_game_time is not None and self.time_seconds is not None:
            if self.time_seconds <= self.cooldown_until_game_time:
                return

        # compute model inputs
        t_frac = (self.time_seconds / self.total_time) if (self.time_seconds is not None and self.total_time) else 0.0
        time_weight_nl = (1.0 - t_frac) ** 2
        score_diff = self.home_score - self.away_score
        home_strength = sum(self.player_weights.get(p, 1.0) for p in self.home_on_court) if self.home_on_court else 0.0
        away_strength = sum(self.player_weights.get(p, 1.0) for p in self.away_on_court) if self.away_on_court else 0.0
        lineup_strength = home_strength - away_strength
        possession_factor = 0.0
        if self.possession == "home":
            possession_factor = 1.0
        elif self.possession == "away":
            possession_factor = -1.0

        # event-derived signals
        run_strength = self._scoring_run_strength(lookback_secs=20.0)   # [-1,1]
        shot_trend = self._shot_distance_trend()
        foul_imb = self._foul_imbalance()

        # Initial offset calibration with small home bias included
        mid_norm = self._mid_normalized()
        if mid_norm is not None:
            if not hasattr(self, "offset") or self.offset is None:
                p_market = min(max(mid_norm, 0.01), 0.99)
                logit = math.log(p_market / (1.0 - p_market))
                model_logit = self.a * score_diff + self.b * time_weight_nl + self.c * possession_factor + self.d * lineup_strength
                # include a small starting home bias (logit space) so at start we slightly favor home
                # This is small by default — tune to 0 to remove prior.
                starting_bias = self.starting_home_bias_logit
                self.offset = logit - (model_logit + starting_bias)
            else:
                # gentle recalibration if market drifts strongly
                if len(self.mid_history) >= max(6, self.mid_history.maxlen // 2):
                    recent_mean = sum(self.mid_history) / len(self.mid_history)
                    if abs(mid_norm - recent_mean) > 0.08 and self.offset is not None:
                        p_market = min(max(mid_norm, 0.01), 0.99)
                        new_logit = math.log(p_market / (1.0 - p_market))
                        model_logit = self.a * score_diff + self.b * time_weight_nl + self.c * possession_factor + self.d * lineup_strength
                        target_offset = new_logit - model_logit
                        alpha = 0.08
                        self.offset = (1.0 - alpha) * self.offset + alpha * target_offset

        # compute model logit and probability (numerically stable)
        x = (self.a * score_diff +
             self.b * time_weight_nl +
             self.c * possession_factor +
             self.d * lineup_strength +
             (self.offset if hasattr(self, "offset") and self.offset is not None else 0.0) +
             0.16 * run_strength +     # run_strength is important
             0.05 * shot_trend +
             0.04 * foul_imb)

        if x >= 0:
            p_home_win = 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            p_home_win = exp_x / (1.0 + exp_x)

        # require market info
        mid_price = self._mid_price()
        if mid_price is None:
            return
        p_market = mid_price / 100.0
        edge = p_home_win - p_market

        # dynamic threshold based on vol & momentum
        dynamic_min_edge = self.min_edge_base + min(0.06, vol * 1.6)
        # if strong run in direction, slightly relax threshold (so we can ride momentum)
        if abs(run_strength) > 0.45:
            dynamic_min_edge *= 0.9
        # disallow trading in blowouts unless massive edge
        if abs(score_diff) >= self.blowout_margin and abs(edge) < 0.06:
            return

        # throttle too-frequent trades by last trade timestamp (game-time)
        if self.last_trade_game_time is not None and self.time_seconds is not None:
            if abs(self.last_trade_game_time - self.time_seconds) < self.min_secs_between_trades:
                return

        # exposure cap
        if self._exposure_fraction() >= self.max_total_exposure:
            return

        # require significant edge
        if abs(edge) <= dynamic_min_edge:
            return

        # confirm with market momentum: don't enter against recent momentum
        mom = mom  # keep variable name consistent; defined earlier via self._momentum
        if mom is None:
            mom = 0.0
        # If momentum is strongly against the model (e.g., model says buy but price momentum is downward), require larger edge
        if edge > 0 and mom < -0.01:
            # price moving down — don't buy unless edge is substantially large
            if edge < dynamic_min_edge * 1.6:
                return
        if edge < 0 and mom > 0.01:
            if abs(edge) < dynamic_min_edge * 1.6:
                return

        # Compute position size: squared-edge / vol heuristic (conservative Kelly intuition) and apply multipliers
        k = 1.1
        raw_frac = k * (edge * edge) / max(vol, self.vol_floor)

        # size up for aligned situation: run + possession + model agree
        if (run_strength > 0.4 and edge > 0 and self.possession == "home") or (run_strength < -0.4 and edge < 0 and self.possession == "away"):
            raw_frac *= 1.5

        # reduce size during high volatility
        if vol > 0.06:
            raw_frac *= 0.45

        # wind down close to end of game
        if self.time_seconds is not None and self.time_seconds <= self.late_game_winddown_secs:
            raw_frac *= 0.45
        if self.time_seconds is not None and self.time_seconds <= self.late_game_cutoff:
            raw_frac *= 0.25

        # clamp to max_fraction and remaining allowance
        frac = min(self.max_fraction, raw_frac)
        remaining_allowance = max(0.0, self.max_total_exposure - self._exposure_fraction())
        frac = min(frac, remaining_allowance)
        if frac <= 0:
            return

        # calculate qty
        if edge > 0:
            # go long (buy home)
            if self.best_ask is None or self.best_ask <= 0:
                return
            qty = (frac * self.capital) / self.best_ask
            if qty < 1e-6:
                return
            # check cooldown_until_game_time
            if self.cooldown_until_game_time is not None and self.time_seconds is not None and self.time_seconds <= self.cooldown_until_game_time:
                return
            self._execute_order(Side.BUY, qty)
            self.last_trade_game_time = self.time_seconds
            self._log(self.time_seconds, event_type, p_home_win, p_market, edge, "BUY", qty, self.best_ask or 0.0,
                      notes=f"run{run_strength:.3f} mom{mom:.4f} vol{vol:.4f}")
        else:
            # go short (sell home)
            if self.best_bid is None or self.best_bid <= 0:
                return
            qty = (frac * self.capital) / self.best_bid
            if qty < 1e-6:
                return
            if self.cooldown_until_game_time is not None and self.time_seconds is not None and self.time_seconds <= self.cooldown_until_game_time:
                return
            self._execute_order(Side.SELL, qty)
            self.last_trade_game_time = self.time_seconds
            self._log(self.time_seconds, event_type, p_home_win, p_market, edge, "SELL", qty, self.best_bid or 0.0,
                      notes=f"run{run_strength:.3f} mom{mom:.4f} vol{vol:.4f}")

        # end-of-game cleanup
        if event_type == "END_GAME":
            # optionally flatten before reset to avoid last-second exposure
            if self.position != 0:
                if self.position > 0:
                    self._execute_order(Side.SELL, abs(self.position))
                else:
                    self._execute_order(Side.BUY, abs(self.position))
            self.reset_state()
