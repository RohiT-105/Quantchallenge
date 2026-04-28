from enum import Enum
from typing import Optional, Dict, List
from collections import deque
import math
import time
import os

# ---------- Exchange Interfaces (placeholder) ----------
class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    TEAM_A = 0  # Home-team win contract

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Platform should implement actual market order."""
    # placeholder
    pass

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Platform should implement actual limit/IOC order. Return order_id if filled, else 0/None."""
    # placeholder
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Platform should implement cancel. Return success boolean."""
    # placeholder
    return False
# -----------------------------------------------------

class Strategy:
    """
    Patched strategy with:
      - wall-clock throttling & cooldowns
      - correct logging with model/market/edge
      - conservative offset recalibration
      - clearer dynamic min-edge & volatility handling
    """

    def reset_state(self) -> None:
        # core game state
        self.home_score = 0
        self.away_score = 0
        self.position = 0.0           # net contracts (positive = long HOME)
        self.capital = 100000.0       # cash bank
        self.best_bid: Optional[float] = None  # price in 0..100 dollars
        self.best_ask: Optional[float] = None
        self.total_time: Optional[float] = None

        # logistic model params
        self.a = 0.30
        self.b = 0.60
        self.c = 0.06
        self.d = 0.02
        self.offset: Optional[float] = None
        self.offset_set = False

        # possession & lineup
        self.possession: Optional[str] = None
        self.player_weights: Dict[str, float] = {f"Player {i}": 1.0 for i in range(1,51)}
        self.home_on_court: List[str] = []
        self.away_on_court: List[str] = []

        # execution & risk knobs
        self.min_edge_base = 0.015
        self.max_fraction = 0.25
        self.max_total_exposure = 0.40
        self.stop_loss_frac = 0.06
        self.take_profit_frac = 0.12
        self.min_secs_between_trades = 2.0
        self.cooldown_after_stop_secs = 5.0
        self.late_game_winddown_secs = 60.0

        # volatility & sizing helpers
        self.volatility_lookback = 20
        self.mid_history = deque(maxlen=self.volatility_lookback)
        self.vol_floor = 1e-4

        # bookkeeping
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

        # timing & throttles (use wall-clock)
        self._last_trade_wall_ts: Optional[float] = None
        self.cooldown_until_wall_ts: Optional[float] = None

        # last game time used for logging (game-clock seconds)
        self.last_trade_game_time: Optional[float] = None

        # logging
        self.log_path = "trades.log"
        if not os.path.exists(self.log_path):
            with open(self.log_path, "a") as f:
                f.write("wall_ts,game_time,event,p_model,p_market,edge,side,qty,price_dollars,position,unrealized,realized\n")

        # safety
        self.start_capital = self.capital

    def __init__(self) -> None:
        self.reset_state()

    # ---- helpers for mid price ----
    def _mid_price_normalized(self) -> Optional[float]:
        """Return mid price normalized to [0,1] (probability) or None if not available."""
        if self.best_bid is None or self.best_ask is None:
            return None
        mid = 0.5 * (self.best_bid + self.best_ask)
        if mid <= 0.0:
            return None
        return mid / 100.0  # normalize to 0..1

    def _mid_price_dollars(self) -> Optional[float]:
        """Return mid price in dollars 0..100 or None."""
        if self.best_bid is None or self.best_ask is None:
            return None
        mid = 0.5 * (self.best_bid + self.best_ask)
        return mid if mid > 0.0 else None

    def _append_mid(self) -> None:
        mid_norm = self._mid_price_normalized()
        if mid_norm is not None:
            self.mid_history.append(mid_norm)

    def _volatility(self) -> float:
        h = list(self.mid_history)
        n = len(h)
        if n < 2:
            return self.vol_floor
        mean = sum(h) / n
        var = sum((x - mean) ** 2 for x in h) / (n - 1)
        return max(math.sqrt(var), self.vol_floor)

    def _exposure_fraction(self) -> float:
        mid_norm = self._mid_price_normalized()
        if mid_norm is None or self.capital <= 0:
            return 0.0
        mid_dollar = mid_norm * 100.0
        exposure_value = abs(self.position) * mid_dollar
        return exposure_value / max(self.capital, 1e-9)

    # ---- logging ----
    def _log_trade(self, wall_ts: float, game_time: Optional[float], event: str,
                   p_model: float, p_market: float, edge: float, side: str, qty: float, price_dollars: float):
        line = ",".join(map(str, [
            round(wall_ts, 6),
            game_time if game_time is not None else "",
            event,
            round(p_model, 6),
            round(p_market, 6),
            round(edge, 6),
            side,
            round(qty, 6),
            round(price_dollars, 6),
            round(self.position, 6),
            round(self.unrealized_pnl, 6),
            round(self.realized_pnl, 6)
        ])) + "\n"
        with open(self.log_path, "a") as f:
            f.write(line)

    # execution wrapper: prefer IOC limit at best price, fallback to market
    def _execute_order(self, side: Side, qty: float, p_model: float, p_market: float, edge: float, game_time: Optional[float]) -> None:
        if qty <= 0:
            return

        price_for_log = 0.0
        try:
            if side == Side.BUY:
                if self.best_ask is not None and self.best_ask > 0:
                    price_for_log = self.best_ask
                    # attempt IOC limit at ask; platform should return a truthy order_id if filled synchronously
                    order_id = place_limit_order(Side.BUY, Ticker.TEAM_A, qty, price=self.best_ask, ioc=True)
                    if not order_id:
                        # fallback to market
                        place_market_order(Side.BUY, Ticker.TEAM_A, qty)
                else:
                    # fallback market
                    price_for_log = 0.0
                    place_market_order(Side.BUY, Ticker.TEAM_A, qty)
            else:
                if self.best_bid is not None and self.best_bid > 0:
                    price_for_log = self.best_bid
                    order_id = place_limit_order(Side.SELL, Ticker.TEAM_A, qty, price=self.best_bid, ioc=True)
                    if not order_id:
                        place_market_order(Side.SELL, Ticker.TEAM_A, qty)
                else:
                    price_for_log = 0.0
                    place_market_order(Side.SELL, Ticker.TEAM_A, qty)
        except Exception:
            # best-effort: fallback to market if limit raises
            if side == Side.BUY:
                place_market_order(Side.BUY, Ticker.TEAM_A, qty)
            else:
                place_market_order(Side.SELL, Ticker.TEAM_A, qty)

        # log the attempted execution with the correct model/market/edge values
        self._log_trade(time.time(), game_time, "EXECUTE", p_model, p_market, edge, side.name, qty, price_for_log or 0.0)

    # -------------------------
    # platform callbacks
    # -------------------------
    def on_trade_update(self, ticker: Ticker, side: Side, price: float, quantity: float) -> None:
        # not used by core logic but keep for interface parity
        return

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        if ticker != Ticker.TEAM_A:
            return
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
        """
        Called when our orders match. Update position, capital, avg_entry_price, realized_pnl and recompute unrealized pnl.
        price is platform fill price (0..100)
        """
        if ticker != Ticker.TEAM_A:
            return

        prev_position = self.position
        self.capital = capital_remaining

        # BUY increases long exposure
        if side == Side.BUY:
            new_position = self.position + quantity
            if prev_position >= 0:
                prev_value = prev_position * self.avg_entry_price
                new_value = quantity * price
                total_qty = prev_position + quantity
                if total_qty > 0:
                    self.avg_entry_price = (prev_value + new_value) / total_qty
                else:
                    self.avg_entry_price = price
            else:
                # we were short; buying reduces short -> realized pnl
                realized = quantity * (self.avg_entry_price - price)
                self.realized_pnl += realized
                if quantity > abs(prev_position):
                    flipped_qty = quantity - abs(prev_position)
                    self.avg_entry_price = price
            self.position = new_position
        else:  # SELL
            new_position = self.position - quantity
            if prev_position <= 0:
                prev_value = abs(prev_position) * self.avg_entry_price
                new_value = quantity * price
                total_qty = abs(prev_position) + quantity
                if total_qty > 0:
                    self.avg_entry_price = (prev_value + new_value) / total_qty
                else:
                    self.avg_entry_price = price
            else:
                realized = quantity * (price - self.avg_entry_price)
                self.realized_pnl += realized
                if quantity > prev_position:
                    flipped_qty = quantity - prev_position
                    self.avg_entry_price = price
            self.position = new_position

        # update unrealized using mid (dollars)
        mid_d = self._mid_price_dollars()
        if mid_d is None:
            self.unrealized_pnl = 0.0
        else:
            self.unrealized_pnl = self.position * (mid_d - self.avg_entry_price)

        # risk checks immediately after fills
        self._post_fill_risk_checks()

    def _post_fill_risk_checks(self) -> None:
        # stop-loss: unrealized negative beyond threshold
        if self.capital <= 0:
            return

        if self.unrealized_pnl < -self.stop_loss_frac * self.capital:
            # flatten immediately
            if self.position > 0:
                self._execute_order(Side.SELL, abs(self.position), 0.0, (self._mid_price_normalized() or 0.0), 0.0, self.last_trade_game_time)
            elif self.position < 0:
                self._execute_order(Side.BUY, abs(self.position), 0.0, (self._mid_price_normalized() or 0.0), 0.0, self.last_trade_game_time)
            # set a short cooldown on wall-clock to avoid immediate re-entry
            self.cooldown_until_wall_ts = time.time() + self.cooldown_after_stop_secs
            return

        # take-profit: partially reduce position
        if self.unrealized_pnl > self.take_profit_frac * self.capital:
            qty = 0.5 * abs(self.position)
            if qty > 0:
                if self.position > 0:
                    self._execute_order(Side.SELL, qty, 0.0, (self._mid_price_normalized() or 0.0), 0.0, self.last_trade_game_time)
                else:
                    self._execute_order(Side.BUY, qty, 0.0, (self._mid_price_normalized() or 0.0), 0.0, self.last_trade_game_time)
                self.cooldown_until_wall_ts = time.time() + self.cooldown_after_stop_secs

    # -------------------------
    # Main game-event logic
    # -------------------------
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
        # update scores & possession
        self.home_score = home_score
        self.away_score = away_score

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

        # total time
        if self.total_time is None and event_type in ("START_PERIOD", "JUMP_BALL") and time_seconds:
            self.total_time = time_seconds
        if self.total_time is None:
            self.total_time = 2880.0

        # stop trading late in game (use game-clock)
        if time_seconds is not None and time_seconds < self.late_game_winddown_secs:
            # still allow position management via on_account_update (stop/take), but avoid new bets
            return

        # stop on drawdown
        if self.capital < (1 - 0.4) * self.start_capital:
            # safety: if -40% drawdown, stop trading altogether
            return

        # append mid history (orderbook callbacks will have pushed mid entries generally)
        self._append_mid()
        vol = self._volatility()

        # model features
        t_frac = time_seconds / self.total_time if (time_seconds is not None and self.total_time) else 0.0
        time_weight_nl = (1.0 - t_frac) ** 2
        score_diff = self.home_score - self.away_score

        home_strength = sum(self.player_weights.get(p, 1.0) for p in self.home_on_court) if self.home_on_court else 0.0
        away_strength = sum(self.player_weights.get(p, 1.0) for p in self.away_on_court) if self.away_on_court else 0.0
        lineup_strength = home_strength - away_strength

        possession_factor = 0.0
        if self.possession == "home":
            possession_factor = 1.0 * time_weight_nl
        elif self.possession == "away":
            possession_factor = -1.0 * time_weight_nl

        # calibrate offset (stable, small learning rate)
        mid_norm = self._mid_price_normalized()
        if mid_norm is not None:
            if not self.offset_set:
                p_market = min(max(mid_norm, 0.01), 0.99)
                logit = math.log(p_market / (1.0 - p_market))
                model_logit = (self.a * score_diff + self.b * time_weight_nl + self.c * possession_factor + self.d * lineup_strength)
                self.offset = logit - model_logit
                self.offset_set = True
            else:
                if len(self.mid_history) >= max(3, self.volatility_lookback // 2):
                    recent_mean = sum(self.mid_history) / len(self.mid_history)
                    if abs(mid_norm - recent_mean) > 0.08 and self.offset is not None:
                        p_market = min(max(mid_norm, 0.01), 0.99)
                        new_logit = math.log(p_market / (1.0 - p_market))
                        model_logit = (self.a * score_diff + self.b * time_weight_nl + self.c * possession_factor + self.d * lineup_strength)
                        target_offset = new_logit - model_logit
                        alpha = 0.03  # small learning rate for stability
                        self.offset = (1.0 - alpha) * self.offset + alpha * target_offset

        # compute model probability stably
        x = (self.a * score_diff + self.b * time_weight_nl + self.c * possession_factor + self.d * lineup_strength + (self.offset if self.offset is not None else 0.0))
        # numerically stable logistic
        if x >= 0:
            p_home_win = 1.0 / (1.0 + math.exp(-x))
        else:
            ex = math.exp(x)
            p_home_win = ex / (1.0 + ex)

        # require market info
        if mid_norm is None:
            return
        p_market = mid_norm
        edge = p_home_win - p_market

        # dynamic minimum edge (explicit and easier to tune)
        dynamic_min_edge = self.min_edge_base + min(0.05, vol * 1.5)
        # scale slightly with late-game (t_frac close to 1) - allow somewhat smaller thresholds late
        dynamic_min_edge *= (0.85 + 0.3 * t_frac)

        # wall-clock cooldown checks
        now = time.time()
        if self._last_trade_wall_ts is not None:
            if now - self._last_trade_wall_ts < self.min_secs_between_trades:
                return
        if self.cooldown_until_wall_ts is not None and now <= self.cooldown_until_wall_ts:
            return

        # exposure cap
        current_exposure = self._exposure_fraction()
        if current_exposure >= self.max_total_exposure:
            return

        # require meaningful edge
        if abs(edge) <= dynamic_min_edge:
            return

        # sizing: squared-edge scaled by 1/vol (conservative)
        k = 1.5
        raw_frac = k * (edge * edge) / max(vol, self.vol_floor)

        # winddown late in game (reduce bet sizes)
        if time_seconds is not None and time_seconds <= self.late_game_winddown_secs:
            raw_frac *= 0.35

        # cap fraction by max and remaining exposure allowance
        frac = min(self.max_fraction, raw_frac)
        remaining_allowance = max(0.0, self.max_total_exposure - current_exposure)
        frac = min(frac, remaining_allowance)

        if frac <= 0:
            return

        # compute qty using price (BUY at ask, SELL at bid). Use dollar price (0..100).
        mid_dollars = self._mid_price_dollars()
        if mid_dollars is None or mid_dollars <= 0:
            return

        # set last_trade timestamps BEFORE execution so logs show them
        self.last_trade_game_time = time_seconds
        self._last_trade_wall_ts = time.time()

        if edge > 0:
            # buy home (go long)
            if self.best_ask is None or self.best_ask <= 0:
                return
            qty = (frac * self.capital) / self.best_ask
            if qty < 1e-9:
                return
            # execute: pass p_model/p_market/edge and game_time for logging
            self._execute_order(Side.BUY, qty, p_home_win, p_market, edge, time_seconds)
        else:
            # sell home (go short)
            if self.best_bid is None or self.best_bid <= 0:
                return
            qty = (frac * self.capital) / self.best_bid
            if qty < 1e-9:
                return
            self._execute_order(Side.SELL, qty, p_home_win, p_market, edge, time_seconds)

        # done: on_account_update will update realized/unrealized when fill happens

        # reset on END_GAME
        if event_type == "END_GAME":
            self.reset_state()

    def on_end_game_cleanup(self) -> None:
        self.reset_state()
