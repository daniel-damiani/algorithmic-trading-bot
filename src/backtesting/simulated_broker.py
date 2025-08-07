"""
Simulated Broker for Backtesting

Mock broker that replicates AlpacaBroker interface but operates on historical data
for backtesting without actual trading or API calls.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from dataclasses import dataclass, field
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SimulatedAccount:
    """Simulated account information"""
    status: str = "ACTIVE"
    equity: float = 10000.0
    cash: float = 10000.0
    buying_power: float = 10000.0
    day_trade_buying_power: float = 10000.0
    portfolio_value: float = 10000.0
    long_market_value: float = 0.0
    short_market_value: float = 0.0


@dataclass
class SimulatedPosition:
    """Simulated position"""
    symbol: str
    qty: float
    side: str  # 'long' or 'short'
    market_value: float
    cost_basis: float
    unrealized_pl: float
    avg_entry_price: float
    current_price: float = 0.0


@dataclass
class SimulatedOrder:
    """Simulated order"""
    id: str
    client_order_id: Optional[str]
    symbol: str
    qty: float
    side: str  # 'buy' or 'sell'
    type: str  # 'market', 'limit', 'stop'
    status: str  # 'new', 'filled', 'canceled'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_price: Optional[float] = None
    filled_qty: float = 0.0
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None


@dataclass
class Trade:
    """Represents a completed trade"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float
    slippage: float
    order_id: str


class SimulatedBroker:
    """
    Simulated broker that mimics AlpacaBroker interface for backtesting
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_per_share: float = 0.005,
        slippage_bps: int = 5
    ):
        """
        Initialize simulated broker
        
        Args:
            initial_capital: Starting capital
            commission_per_share: Commission cost per share
            slippage_bps: Slippage in basis points
        """
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_bps = slippage_bps / 10000.0  # Convert bps to decimal
        
        # Account state
        self.account = SimulatedAccount(
            equity=initial_capital,
            cash=initial_capital,
            buying_power=initial_capital,
            portfolio_value=initial_capital
        )
        
        # Positions and orders
        self.positions: Dict[str, SimulatedPosition] = {}
        self.open_orders: Dict[str, SimulatedOrder] = {}
        self.filled_orders: Dict[str, SimulatedOrder] = {}
        
        # Historical data and current time for simulation
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_time: Optional[datetime] = None
        
        # Trade and equity tracking
        self.trade_history: List[Trade] = []
        self.equity_history: List[Dict[str, Any]] = []
        
        # Risk management tracking
        self._daily_start_equity = initial_capital
        self._current_date = None
        
        # Connection state
        self.is_connected = False
        
        logger.info("Simulated broker initialized",
                   initial_capital=initial_capital,
                   commission_per_share=commission_per_share,
                   slippage_bps=slippage_bps * 10000)
    
    def set_historical_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Set historical data for simulation"""
        self.historical_data = data
        logger.info(f"Historical data loaded for {len(data)} symbols")
    
    def set_current_time(self, timestamp: datetime) -> None:
        """Update current simulation time and handle daily resets"""
        self.current_time = timestamp
        
        # Check if we've moved to a new trading day
        current_date = timestamp.date()
        if self._current_date is None:
            self._current_date = current_date
            self._daily_start_equity = self.equity
        elif self._current_date != current_date:
            # New trading day - reset daily tracking
            self._current_date = current_date
            self._daily_start_equity = self.equity
            logger.debug("New trading day", date=current_date, starting_equity=self.equity)
        
        self._update_positions_value()
        self._record_equity_snapshot()
    
    async def connect(self) -> bool:
        """Simulate connection"""
        self.is_connected = True
        logger.info("Simulated broker connected")
        return True
    
    async def disconnect(self) -> None:
        """Simulate disconnection"""
        self.is_connected = False
        logger.info("Simulated broker disconnected")
    
    # === ACCOUNT METHODS ===
    
    async def get_account(self) -> SimulatedAccount:
        """Get simulated account information"""
        return self.account
    
    async def get_buying_power(self) -> float:
        """Get available buying power"""
        return self.account.buying_power
    
    async def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        return self.account.portfolio_value
    
    # === POSITION METHODS ===
    
    async def get_positions(self) -> List[SimulatedPosition]:
        """Get all current positions"""
        return list(self.positions.values())
    
    async def get_position(self, symbol: str) -> Optional[SimulatedPosition]:
        """Get position for specific symbol"""
        return self.positions.get(symbol)
    
    # === DATA METHODS ===
    
    async def get_bars(
        self,
        symbol: str,
        timeframe: str = '1Hour',
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get historical bars for simulation
        Returns data up to current simulation time
        """
        if symbol not in self.historical_data:
            logger.warning(f"Symbol {symbol} not in historical data")
            return None
        
        data = self.historical_data[symbol].copy()
        
        # Filter data up to current simulation time (use end parameter if provided, otherwise current_time)
        filter_time = end if end is not None else self.current_time
        if filter_time:
            data = data[data.index <= filter_time]
            logger.debug(f"Filtered data to {filter_time}, remaining: {len(data)} bars")
        
        # Apply date filters
        if start:
            data = data[data.index >= start]
            logger.debug(f"Applied start filter {start}, remaining: {len(data)} bars")
        
        # Apply limit - but make sure we return data even if less than limit
        if limit and len(data) > limit:
            # Get the last 'limit' bars up to the current time
            data = data.tail(limit)
            logger.debug(f"Applied limit {limit}, returning last {limit} bars")
        elif limit:
            # If we have less data than limit, still return what we have
            logger.debug(f"Requested {limit} bars, but only {len(data)} available")
        
        if data.empty:
            logger.info(f"No data for {symbol} at time {filter_time}")
            return None
            
        logger.debug(f"Returning {len(data)} bars for {symbol} from {data.index.min()} to {data.index.max()}")
        return data
    
    async def get_latest_quote(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get latest quote for symbol at current simulation time"""
        if symbol not in self.historical_data or not self.current_time:
            return None
        
        data = self.historical_data[symbol]
        
        # Find latest data point at or before current time
        valid_data = data[data.index <= self.current_time]
        if valid_data.empty:
            return None
        
        latest_bar = valid_data.iloc[-1]
        
        return {
            'bid': latest_bar['close'],
            'ask': latest_bar['close'],
            'last': latest_bar['close'],
            'timestamp': latest_bar.name
        }
    
    def is_market_open(self) -> bool:
        """Simulate market hours - assume always open for simplicity"""
        return True
    
    # === ORDER METHODS ===
    
    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: Union[int, float],
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None
    ) -> SimulatedOrder:
        """
        Submit simulated order
        """
        # Create order
        order_id = str(uuid.uuid4())
        order = SimulatedOrder(
            id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            qty=float(quantity),
            side=side,
            type=order_type,
            status="new",
            limit_price=limit_price,
            stop_price=stop_price,
            submitted_at=self.current_time
        )
        
        self.open_orders[order_id] = order
        
        # For market orders, fill immediately
        if order_type.lower() == "market":
            await self._fill_order(order)
        
        logger.info("Simulated order submitted",
                   order_id=order_id,
                   symbol=symbol,
                   side=side,
                   quantity=quantity,
                   order_type=order_type)
        
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel simulated order"""
        if order_id in self.open_orders:
            order = self.open_orders[order_id]
            order.status = "canceled"
            del self.open_orders[order_id]
            logger.info("Order canceled", order_id=order_id)
            return True
        return False
    
    async def get_orders(self, status: str = "all", limit: int = 100) -> List[SimulatedOrder]:
        """Get simulated orders"""
        if status == "open":
            orders = list(self.open_orders.values())
        elif status == "filled":
            orders = list(self.filled_orders.values())
        else:
            orders = list(self.open_orders.values()) + list(self.filled_orders.values())
        
        return orders[:limit]
    
    async def modify_order(
        self,
        order_id: str,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> bool:
        """Modify simulated order"""
        if order_id not in self.open_orders:
            return False
        
        order = self.open_orders[order_id]
        
        if quantity is not None:
            order.qty = quantity
        if limit_price is not None:
            order.limit_price = limit_price
        if stop_price is not None:
            order.stop_price = stop_price
        
        logger.info("Order modified", order_id=order_id)
        return True
    
    # === SIMULATION METHODS ===
    
    async def _fill_order(self, order: SimulatedOrder) -> None:
        """Fill a simulated order"""
        # Get current price
        quote = await self.get_latest_quote(order.symbol)
        if not quote:
            logger.warning(f"No price data available for {order.symbol}, cannot fill order")
            return
        
        # Calculate fill price with slippage
        base_price = quote['last']
        slippage_factor = self.slippage_bps if order.side == 'buy' else -self.slippage_bps
        fill_price = base_price * (1 + slippage_factor)
        
        # Calculate commission
        commission = order.qty * self.commission_per_share
        
        # Check if we have enough buying power/shares
        if order.side == 'buy':
            required_capital = (order.qty * fill_price) + commission
            if required_capital > self.account.cash:
                logger.warning(f"Insufficient cash for order {order.id}")
                order.status = "rejected"
                return
        else:  # sell
            position = self.positions.get(order.symbol)
            if not position or position.qty < order.qty:
                logger.warning(f"Insufficient shares for sell order {order.id}")
                order.status = "rejected"
                return
        
        # Fill the order
        order.status = "filled"
        order.filled_price = fill_price
        order.filled_qty = order.qty
        order.filled_at = self.current_time
        
        # Update positions and account
        await self._update_position(order, fill_price, commission)
        
        # Move to filled orders
        self.filled_orders[order.id] = order
        if order.id in self.open_orders:
            del self.open_orders[order.id]
        
        # Record trade
        trade = Trade(
            timestamp=self.current_time,
            symbol=order.symbol,
            side=order.side,
            quantity=order.qty,
            price=fill_price,
            commission=commission,
            slippage=abs(fill_price - base_price) * order.qty,
            order_id=order.id
        )
        self.trade_history.append(trade)
        
        logger.info("Order filled",
                   order_id=order.id,
                   symbol=order.symbol,
                   side=order.side,
                   quantity=order.qty,
                   price=fill_price,
                   commission=commission)
    
    async def _update_position(
        self,
        order: SimulatedOrder,
        fill_price: float,
        commission: float
    ) -> None:
        """Update position based on filled order"""
        symbol = order.symbol
        
        if order.side == 'buy':
            if symbol in self.positions:
                # Add to existing position
                pos = self.positions[symbol]
                total_cost = (pos.qty * pos.avg_entry_price) + (order.qty * fill_price)
                total_qty = pos.qty + order.qty
                pos.avg_entry_price = total_cost / total_qty
                pos.qty = total_qty
                pos.cost_basis = total_cost
            else:
                # Create new position
                self.positions[symbol] = SimulatedPosition(
                    symbol=symbol,
                    qty=order.qty,
                    side='long',
                    market_value=order.qty * fill_price,
                    cost_basis=order.qty * fill_price,
                    unrealized_pl=0.0,
                    avg_entry_price=fill_price,
                    current_price=fill_price
                )
            
            # Update cash
            self.account.cash -= (order.qty * fill_price) + commission
        
        else:  # sell
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.qty -= order.qty
                
                # If position closed, remove it
                if pos.qty <= 0:
                    del self.positions[symbol]
                
                # Update cash
                self.account.cash += (order.qty * fill_price) - commission
    
    def _update_positions_value(self) -> None:
        """Update position values based on current prices"""
        total_position_value = 0.0
        
        for symbol, position in self.positions.items():
            quote = None
            if symbol in self.historical_data and self.current_time:
                data = self.historical_data[symbol]
                valid_data = data[data.index <= self.current_time]
                if not valid_data.empty:
                    position.current_price = valid_data.iloc[-1]['close']
                    position.market_value = position.qty * position.current_price
                    position.unrealized_pl = position.market_value - position.cost_basis
                    total_position_value += position.market_value
        
        # Update account values
        self.account.long_market_value = total_position_value
        self.account.portfolio_value = self.account.cash + total_position_value
        self.account.equity = self.account.portfolio_value
        self.account.buying_power = self.account.cash  # Simplified
    
    def _record_equity_snapshot(self) -> None:
        """Record current equity for performance tracking"""
        if self.current_time:
            self.equity_history.append({
                'timestamp': self.current_time,
                'equity': self.account.equity,
                'cash': self.account.cash,
                'positions_value': self.account.long_market_value
            })
    
    @property
    def equity(self) -> float:
        """Current portfolio equity"""
        return self.account.equity