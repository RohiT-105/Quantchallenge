from enum import Enum
from typing import Optional
import math

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    TEAM_A = 0  # The home-team win contract

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    return False

class Strategy:
    def reset_state(self) -> None:
        # Reset at the start of each game
        self.home_score = 0
        self.away_score = 0
        self.position = 0.0
        self.capital = 100000.0  # initial bankroll
        self.best_bid = None
        self.best_ask = None
        # Model parameters (tunables)
        self.a = 0.3   # weight on score differential
        self.b = 0.2   # weight on time remaining
        self.offset = None  # bias term to align with market odds
        self.total_time = None

    def __init__(self) -> None:
        self.reset_state()

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        # Maintain best bid/ask from incremental updates
        if ticker == Ticker.TEAM_A:
            if side == Side.BUY:
                if quantity > 0:
                    if self.best_bid is None or price > self.best_bid:
                        self.best_bid = price
                elif self.best_bid == price:
                    self.best_bid = None
            elif side == Side.SELL:
                if quantity > 0:
                    if self.best_ask is None or price < self.best_ask:
                        self.best_ask = price
                elif self.best_ask == price:
                    self.best_ask = None

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        if ticker == Ticker.TEAM_A:
            # bids and asks are sorted lists of (price, quantity)
            if bids:
                self.best_bid = bids[0][0]
            if asks:
                self.best_ask = asks[0][0]

    def on_account_update(self,
        ticker: Ticker,
        side: Side,
        price: float,
        quantity: float,
        capital_remaining: float,
    ) -> None:
        # Update our inventory and cash after our orders fill
        if ticker == Ticker.TEAM_A:
            if side == Side.BUY:
                self.position += quantity
            elif side == Side.SELL:
                self.position -= quantity
            self.capital = capital_remaining

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
        # Update scores from the event data
        self.home_score = home_score
        self.away_score = away_score

        # Determine total game time at the start of the game
        if self.total_time is None and event_type in ("JUMP_BALL", "START_PERIOD"):
            self.total_time = time_seconds

        # If we still don't know total_time (mid-game start), assume 2880 sec
        if self.total_time is None:
            self.total_time = 2880.0

        # Calculate fraction of game time elapsed
        t_frac = time_seconds / self.total_time if self.total_time else 0.0
        score_diff = home_score - away_score

        # Calibrate model offset once using the initial market mid-price
        if (self.offset is None 
                and self.best_bid is not None and self.best_ask is not None):
            mid_price = (self.best_bid + self.best_ask) / 2.0
            if mid_price > 0:
                p_market = mid_price / 100.0
                p_market = max(min(p_market, 0.99), 0.01)
                # Solve for offset so that model probability = market probability
                self.offset = math.log(p_market / (1 - p_market)) - (self.a * score_diff + self.b * t_frac)
        if self.offset is None:
            self.offset = 0.0  # fallback if calibration didn't run

        # Compute home win probability using logistic model
        x = self.a * score_diff + self.b * t_frac + self.offset
        p_home_win = 1.0 / (1.0 + math.exp(-x))

        # Determine market-implied probability from mid-price
        if self.best_bid is not None and self.best_ask is not None:
            market_mid = (self.best_bid + self.best_ask) / 2.0
            market_prob = market_mid / 100.0

            # If our model gives higher P(home win), buy home-win contracts
            if p_home_win > market_prob and self.best_ask is not None:
                edge = p_home_win - market_prob
                frac = min(0.5, edge * 2.0)  # scale factor for bet size
                if frac > 0 and self.capital > 0 and self.best_ask > 0:
                    qty = (frac * self.capital) / self.best_ask
                    place_market_order(Side.BUY, Ticker.TEAM_A, qty)

            # If our model gives lower P(home win), buy away-win (sell home-win)
            elif p_home_win < market_prob and self.best_bid is not None:
                edge = market_prob - p_home_win
                frac = min(0.5, edge * 2.0)
                if frac > 0 and self.capital > 0 and self.best_bid > 0:
                    qty = (frac * self.capital) / self.best_bid
                    place_market_order(Side.SELL, Ticker.TEAM_A, qty)

        # Reset at game end
        if event_type == "END_GAME":
            self.reset_state()
            return
