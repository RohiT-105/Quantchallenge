from enum import Enum
from typing import Optional, Dict, List
from collections import deque
import math
import time
import json
import os

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    TEAM_A = 0  # Home-team win contract

# --- Placeholder exchange functions: replace with platform implementations on submit ---
def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    """Platform should implement actual market order."""
    pass

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    """Platform should implement actual limit/IOC order. Return order_id if applicable."""
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    """Platform should implement cancel. Return success boolean."""
    return False
# -------------------------------------------------------------------------------------

class Strategy:
    """
    Final improved strategy:
      - Logistic in-game win probability using score diff, nonlinear time weight, possession, lineup strength
      - Risk controls: stop-loss, take-profit, exposure cap, cooldown, late-game winddown
      - Volatility-aware sizing, IOC-limit wrapper for execution, conservative recalibration
      - Per-trade logging to trades.log
    """

    def reset_state(self) -> None:
        # -------------------------
        # Core game state
        # -------------------------
        self.home_score = 0
        self.away_score = 0
        self.position = 0.0           # net contracts (positive = long HOME)
        self.capital = 100000.0       # cash bank
        self.best_bid: Optional[float] = None  # price in [0,100]
        self.best_ask: Optional[float] = None
        self.total_time: Optional[float] = None
        self.offset: Optional[float] = None
        self.offset_set = False

        # -------------------------
        # Model weights (leave conservative)
        # -------------------------
        self.a = 0.30   # score differential weight
        self.b = 0.60   # base time weight magnitude (applied nonlinearly)
        self.c = 0.06   # possession weight
        self.d = 0.02   # lineup strength weight

        # possession & lineup
        self.possession: Optional[str] = None
        self.player_weights: Dict[str, float] = {f"Player {i}": 1.0 for i in range(1,51)}
        self.home_on_court: List[str] = []
        self.away_on_court: List[str] = []

        # -------------------------
        # Execution & risk knobs (tune conservatively)
        # -------------------------
        self.min_edge_base = 0.015       # base minimum probability edge to consider trading
        self.max_fraction = 0.25         # max fraction of capital for a single trade
        self.max_total_exposure = 0.40   # cap: total exposure fraction of capital
        self.stop_loss_frac = 0.06       # flatten if unrealized loss > 6% of capital
        self.take_profit_frac = 0.12     # take partial profits if unrealized gain > 12% of capital
        self.min_secs_between_trades = 2.0
        self.late_game_winddown_secs = 60.0  # in final X seconds reduce sizing

        # -------------------------
        # Volatility & sizing helpers
        # -------------------------
        self.volatility_lookback = 20
        self.mid_history = deque(maxlen=self.volatility_lookback)
        self.vol_floor = 1e-4

        # -------------------------
        # Trade bookkeeping
        # -------------------------
        self.avg_entry_price = 0.0     # dollar price per contract (0..100) for net position
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.last_trade_game_time: Optional[float] = None
        self.cooldown_until_game_time: Optional[float] = None

        # logging
        self.log_path = "trades.log"
        # create/append header on new file
        if not os.path.exists(self.log_path):
            with open(self.log_path, "a") as f:
                f.write("time,game_time,event,p_model,p_market,edge,side,qty,price,position,unrealized,realized\n")

    def __init__(self) -> None:
        self.reset_state()

    # -------------------------
    # Low-level helpers
    # -------------------------
    def _mid_price_normalized(self) -> Optional[float]:
        """Return mid price normalized to [0,1] or None if not available."""
        if self.best_bid is None or self.best_ask is None:
            return None
        mid = 0.5 * (self.best_bid + self.best_ask)
        if mid <= 0.0:
            return None
        return mid / 100.0

    def _append_mid(self) -> None:
        mid = self._mid_price_normalized()
        if mid is not None:
            self.mid_history.append(mid)

    def _volatility(self) -> float:
        h = list(self.mid_history)
        n = len(h)
        if n < 2:
            return self.vol_floor
        mean = sum(h) / n
        var = sum((x - mean) ** 2 for x in h) / (n - 1)
        return max(math.sqrt(var), self.vol_floor)

    def _exposure_fraction(self) -> float:
        """Estimate exposure as |position| * mid_price * 100 / capital."""
        mid = self._mid_price_normalized()
        if mid is None or self.capital <= 0:
            return 0.0
        exposure_value = abs(self.position) * mid * 100.0
        return exposure_value / max(self.capital, 1e-9)

    def _log_trade(self, game_time: Optional[float], event: str, p_model: float, p_market: float,
                   edge: float, side: str, qty: float, price: float):
        # append CSV line
        line = ",".join(map(str, [
            time.time(),                    # wall-clock epoch
            game_time if game_time is not None else "",
            event,
            round(p_model,6),
            round(p_market,6),
            round(edge,6),
            side,
            round(qty,6),
            round(price,6),
            round(self.position,6),
            round(self.unrealized_pnl,6),
            round(self.realized_pnl,6)
        ])) + "\n"
        with open(self.log_path, "a") as f:
            f.write(line)

    # Execution wrapper: prefer IOC limit at best price, fallback to market
    def _execute_order(self, side: Side, qty: float) -> None:
        if qty <= 0:
            return
        # price used for logging (best available side)
        price_for_log = None
        if side == Side.BUY:
            if self.best_ask is not None and self.best_ask > 0:
                price_for_log = self.best_ask
                # try IOC limit at ask
                try:
                    order_id = place_limit_order(Side.BUY, Ticker.TEAM_A, qty, price=self.best_ask, ioc=True)
                    # if platform fills via on_account_update, that's handled there
                    if not order_id:
                        # placeholder platform: no id returned -> fall back to market
                        place_market_order(Side.BUY, Ticker.TEAM_A, qty)
                except Exception:
                    place_market_order(Side.BUY, Ticker.TEAM_A, qty)
            else:
                price_for_log = 0.0
                place_market_order(Side.BUY, Ticker.TEAM_A, qty)
        else:
            if self.best_bid is not None and self.best_bid > 0:
                price_for_log = self.best_bid
                try:
                    order_id = place_limit_order(Side.SELL, Ticker.TEAM_A, qty, price=self.best_bid, ioc=True)
                    if not order_id:
                        place_market_order(Side.SELL, Ticker.TEAM_A, qty)
                except Exception:
                    place_market_order(Side.SELL, Ticker.TEAM_A, qty)
            else:
                price_for_log = 0.0
                place_market_order(Side.SELL, Ticker.TEAM_A, qty)

        # logging: approximate log now; exact realized/unrealized will be updated in on_account_update
        self._log_trade(self.last_trade_game_time, "EXECUTE", 0.0, (self._mid_price_for_log() or 0.0), 0.0,
                        side.name, qty, price_for_log or 0.0)

    def _mid_price_for_log(self) -> Optional[float]:
        mid = None
        if self.best_bid is not None and self.best_ask is not None:
            mid = 0.5 * (self.best_bid + self.best_ask)
        return mid

    # -------------------------
    # Required callback stubs
    # -------------------------
    def on_trade_update(self, ticker: Ticker, side: Side, price: float, quantity: float) -> None:
        # provided for linting; not used for logic
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
        Called when one of our orders matches. Update position, capital, avg_entry_price,
        realized_pnl and recompute unrealized pnl.
        price is platform fill price (0..100)
        """
        if ticker != Ticker.TEAM_A:
            return

        prev_position = self.position
        # update capital
        self.capital = capital_remaining
        # Update position & avg entry price carefully:
        if side == Side.BUY:
            # buying increases long exposure
            new_position = self.position + quantity
            # update avg entry price (dollar-weighted)
            if prev_position >= 0:
                # previously long or flat -> weighted average
                prev_value = prev_position * self.avg_entry_price
                new_value = quantity * price
                total_qty = prev_position + quantity
                if total_qty > 0:
                    self.avg_entry_price = (prev_value + new_value) / total_qty
                else:
                    self.avg_entry_price = price
            else:
                # we were short; buying reduces short (may flip)
                # realized pnl from reducing the short
                realized = quantity * (self.avg_entry_price - price)
                self.realized_pnl += realized
                if abs(quantity) > abs(prev_position):
                    # flipped to net long: new avg based on remaining
                    flipped_qty = quantity - abs(prev_position)
                    self.avg_entry_price = price  # cost basis for new long portion
            self.position = new_position
        else:  # SELL
            new_position = self.position - quantity
            if prev_position <= 0:
                # previously short or flat -> update avg_entry_price for short side (store price as cost basis)
                prev_value = abs(prev_position) * self.avg_entry_price
                new_value = quantity * price
                total_qty = abs(prev_position) + quantity
                if total_qty > 0:
                    self.avg_entry_price = (prev_value + new_value) / total_qty
                else:
                    self.avg_entry_price = price
            else:
                # we were long; selling reduces long (realize pnl)
                realized = quantity * (price - self.avg_entry_price)
                self.realized_pnl += realized
                if quantity > prev_position:
                    # flipped to net short
                    flipped_qty = quantity - prev_position
                    self.avg_entry_price = price
            self.position = new_position

        # update unrealized
        self._update_unrealized()

        # enforce immediate stop/take actions if thresholds reached
        self._post_fill_risk_checks()

    # helper to compute unrealized pnl
    def _update_unrealized(self) -> None:
        mid = self._mid_price_for_log()
        if mid is None:
            self.unrealized_pnl = 0.0
            return
        mid_dollar = mid  # mid in 0..100 dollars per contract
        # exposure value difference between current mid and average entry
        self.unrealized_pnl = self.position * (mid_dollar - self.avg_entry_price)

    def _post_fill_risk_checks(self) -> None:
        # check stop-loss / take-profit right after fills
        if self.capital <= 0:
            return
        # stop-loss: unrealized negative beyond threshold
        if self.unrealized_pnl < -self.stop_loss_frac * self.capital:
            # flatten immediately
            if self.position > 0:
                self._execute_order(Side.SELL, abs(self.position))
            elif self.position < 0:
                self._execute_order(Side.BUY, abs(self.position))
            # cooldown for a short while (use game time if available)
            self.cooldown_until_game_time = (self.last_trade_game_time - 0.0) if self.last_trade_game_time is not None else None
            return

        # take-profit: partially reduce position
        if self.unrealized_pnl > self.take_profit_frac * self.capital:
            # partial take: reduce half position
            qty = 0.5 * abs(self.position)
            if qty > 0:
                if self.position > 0:
                    self._execute_order(Side.SELL, qty)
                else:
                    self._execute_order(Side.BUY, qty)
                self.cooldown_until_game_time = (self.last_trade_game_time - 0.0) if self.last_trade_game_time is not None else None

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
        # Update core state
        self.home_score = home_score
        self.away_score = away_score

        # possession heuristic
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

        # establish total time if available
        if self.total_time is None and event_type in ("START_PERIOD", "JUMP_BALL") and time_seconds:
            self.total_time = time_seconds
        if self.total_time is None:
            # fallback (supports mid-game join default)
            self.total_time = 2880.0

        # time features
        t_frac = time_seconds / self.total_time if (time_seconds is not None and self.total_time) else 0.0
        # nonlinear: amplify late game -> squared
        time_weight_nl = (1.0 - t_frac) ** 2
        score_diff = self.home_score - self.away_score

        # lineup strength (simple additive)
        home_strength = sum(self.player_weights.get(p,1.0) for p in self.home_on_court) if self.home_on_court else 0.0
        away_strength = sum(self.player_weights.get(p,1.0) for p in self.away_on_court) if self.away_on_court else 0.0
        lineup_strength = home_strength - away_strength

        # possession factor scaled by late-game sensitivity
        possession_factor = 0.0
        if self.possession == "home":
            possession_factor = 1.0 * time_weight_nl
        elif self.possession == "away":
            possession_factor = -1.0 * time_weight_nl

        # calibrate offset initially or optionally nudge it conservatively if drift
        mid = self._mid_price_normalized()
        if mid is not None:
            if not self.offset_set:
                p_market = min(max(mid, 0.01), 0.99)
                logit = math.log(p_market / (1.0 - p_market))
                self.offset = logit - (self.a * score_diff + self.b * time_weight_nl + self.c * possession_factor + self.d * lineup_strength)
                self.offset_set = True
            else:
                # conservative recalibration if mid-price drift large vs history
                if len(self.mid_history) >= self.volatility_lookback // 2:
                    recent_mean = sum(self.mid_history) / len(self.mid_history)
                    if abs(mid - recent_mean) > 0.08 and self.offset is not None:
                        # small learning rate alpha ensures stability
                        p_market = min(max(mid, 0.01), 0.99)
                        new_logit = math.log(p_market / (1.0 - p_market))
                        model_logit = (self.a * score_diff + self.b * time_weight_nl + self.c * possession_factor + self.d * lineup_strength)
                        target_offset = new_logit - model_logit
                        alpha = 0.10
                        self.offset = (1.0 - alpha) * self.offset + alpha * target_offset

        # compute model probability (numerically stable)
        x = (self.a * score_diff + self.b * time_weight_nl + self.c * possession_factor + self.d * lineup_strength + (self.offset if self.offset is not None else 0.0))
        if x >= 0:
            p_home_win = 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            p_home_win = exp_x / (1.0 + exp_x)

        # update mid history and vol
        self._append_mid()
        vol = self._volatility()

        # if we don't have market info, skip trading
        if mid is None:
            return

        p_market = mid
        edge = p_home_win - p_market

        # dynamic minimum edge increases with volatility (be more conservative when noisy)
        dynamic_min_edge = self.min_edge_base + min(0.05, vol * 1.5)
        # slight late-game relaxation to allow decisive bets (we allow somewhat smaller edges late)
        dynamic_min_edge *= (0.85 + 0.3 * (1.0 - (1.0 - t_frac)))  # ~0.85..1.15 scaling conservative

        # throttle: cooldown or min secs between trades
        if self.last_trade_game_time is not None and time_seconds is not None:
            elapsed = abs((self.last_trade_game_time - time_seconds))
        else:
            elapsed = float('inf')
        if elapsed < self.min_secs_between_trades:
            return

        # enforce exposure cap
        current_exposure = self._exposure_fraction()
        if current_exposure >= self.max_total_exposure:
            return

        # require edge > threshold
        if abs(edge) <= dynamic_min_edge:
            return

        # compute raw fraction: squared-edge scaled by 1/vol
        k = 1.5
        raw_frac = k * (edge * edge) / max(vol, self.vol_floor)

        # winddown late in game
        if time_seconds is not None and time_seconds <= self.late_game_winddown_secs:
            raw_frac *= 0.35  # reduce bet size sharply in the final window

        # cap by max_fraction and by remaining exposure allowance
        frac = min(self.max_fraction, raw_frac)
        remaining_allowance = max(0.0, self.max_total_exposure - current_exposure)
        frac = min(frac, remaining_allowance)

        if frac <= 0:
            return

        # compute qty using price (BUY at ask, SELL at bid)
        if edge > 0:
            # want to buy home-win (go long)
            if self.best_ask is None or self.best_ask <= 0:
                return
            qty = (frac * self.capital) / self.best_ask
            # sanity floor
            if qty < 1e-6:
                return
            # check cooldown_until_game_time
            if self.cooldown_until_game_time is not None and time_seconds is not None and time_seconds <= self.cooldown_until_game_time:
                return
            # execute via IOC-limit wrapper
            self._execute_order(Side.BUY, qty)
            # record last_trade_game_time for throttling/logging
            self.last_trade_game_time = time_seconds
            # log trade (approx)
            self._log_trade(time_seconds, event_type, p_home_win, p_market, edge, "BUY", qty, self.best_ask or 0.0)
        else:
            # want to sell home-win (go short)
            if self.best_bid is None or self.best_bid <= 0:
                return
            qty = (frac * self.capital) / self.best_bid
            if qty < 1e-6:
                return
            if self.cooldown_until_game_time is not None and time_seconds is not None and time_seconds <= self.cooldown_until_game_time:
                return
            self._execute_order(Side.SELL, qty)
            self.last_trade_game_time = time_seconds
            self._log_trade(time_seconds, event_type, p_home_win, p_market, edge, "SELL", qty, self.best_bid or 0.0)

        # early return; on_account_update will handle realized/unrealized updates when fills occur

        # Reset state at END_GAME if that event arrives
        if event_type == "END_GAME":
            self.reset_state()

    # Optional explicit cleanup (some platforms use END_GAME via on_game_event_update)
    def on_end_game_cleanup(self) -> None:
        self.reset_state()
