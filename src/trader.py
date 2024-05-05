# Armaan Kapoor - https://github.com/ak2k2
# IMC Prosperity 2024 - Team Quant NYC
# Members: Armaan Kapoor, Harshil Cherukuri and Shyam Parikh

import collections
import json
import math
import statistics
import string
from typing import Any, List

import jsonpickle
import numpy as np
import pandas as pd

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
    UserId,
)


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing["symbol"], listing["product"], listing["denomination"]]
            )

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

POSITION_LIMITS = {
    "AMETHYSTS": 20,
    "STARFRUIT": 20,
    "ORCHIDS": 100,
    "CHOCOLATE": 250,
    "STRAWBERRIES": 350,
    "ROSES": 60,
    "GIFT_BASKET": 60,
    "COCONUT": 300,
    "COCONUT_COUPON": 600,
}

AMETHYSTS_MEAN = 10_000
STARFRUIT_COEFFICIENTS = [5.24986188, 0.70354115, 0.23410216, 0.04909509, 0.01222407]
CACHE_LIMIT = 1000

np.random.seed(69420)


class Trader:
    def __init__(self):
        # ROUND 1
        self.starfruit_vwap_cache = []
        # ROUND 2
        self.orchids_vwap_cache = []
        # ROUND 3
        self.chocolate_midprice_cache = []
        self.strawberries_midprice_cache = []
        self.roses_midprice_cache = []
        self.gift_basket_midprice_cache = []
        self.arb_basket_cache = []
        # ROUND 4
        self.coconut_midprice_cache = []
        self.coconut_coupon_midprice_cache = []
        # Z_SCORES
        self.chocolate_q_zscore = 0
        self.roses_q_zscore = 0
        self.gift_basket_q_zscore = 0
        self.strawberries_lr_res_q_zscore = 0
        self.coconut_divergence_zscore = 0
        # BS COCONUTS
        self.bs_call_price_cache = []
        self.coconut_coupon_bsm_diff_z_score = 0
        self.order_data_cache = []
        self.vlad_pos = 0

    def deserialize_trader_data(self, state_data):
        try:
            return jsonpickle.decode(state_data)
        except:
            return {}

    def serialize_trader_data(self, data):
        try:
            return jsonpickle.encode(data)
        except:
            return None

    def update_product_price_cache_listing(
        self, previous_trading_state, state: TradingState
    ):
        cache_listing = [
            ("ORCHIDS", "orchids_vwap_cache", 4),
            ("STARFRUIT", "starfruit_vwap_cache", 4),
            ("CHOCOLATE", "chocolate_midprice_cache", CACHE_LIMIT),
            ("STRAWBERRIES", "strawberries_midprice_cache", CACHE_LIMIT),
            ("ROSES", "roses_midprice_cache", CACHE_LIMIT),
            ("GIFT_BASKET", "gift_basket_midprice_cache", CACHE_LIMIT),
            ("COCONUT", "coconut_midprice_cache", CACHE_LIMIT),
            ("COCONUT_COUPON", "coconut_coupon_midprice_cache", CACHE_LIMIT),
        ]
        for product, cache_name, memory in cache_listing:
            cache_hits = previous_trading_state.get(cache_name, [])
            orders = state.order_depths.get(product, OrderDepth())
            sell_orders = orders.sell_orders
            buy_orders = orders.buy_orders

            # VWAP CACHE UPDATES
            if cache_name in ["starfruit_vwap_cache", "orchids_vwap_cache"]:
                if sell_orders and buy_orders:
                    sell_vwap = self.vwap(sell_orders)
                    buy_vwap = self.vwap(buy_orders)
                    current_vwap = (sell_vwap + buy_vwap) / 2
                    cache_hits.append(current_vwap)
                    cache_hits = cache_hits[
                        -memory:
                    ]  # Keep only the last 'memory' entries
                setattr(self, cache_name, cache_hits)

            # MIDPRICE CACHE UPDATES
            elif cache_name in [
                "chocolate_midprice_cache",
                "strawberries_midprice_cache",
                "roses_midprice_cache",
                "gift_basket_midprice_cache",
                "coconut_midprice_cache",
                "coconut_coupon_midprice_cache",
            ]:
                if sell_orders and buy_orders:
                    best_bid_price = max(buy_orders.keys())
                    best_ask_price = min(sell_orders.keys())
                    midprice = (best_bid_price + best_ask_price) / 2
                    cache_hits.append(midprice)
                    cache_hits = cache_hits[
                        -memory:
                    ]  # Keep only the last 'memory' entries
                setattr(self, cache_name, cache_hits)

    def update_black_scholes_cache(self, previous_trading_state, state):
        current_timestamp = state.timestamp
        cache_hits = previous_trading_state.get("bs_call_price_cache", [])

        if self.coconut_midprice_cache:
            timestamps_to_expiration = 246_000_000 - current_timestamp
            days_to_expiration = timestamps_to_expiration / 1_000_000
            cache_hits.append(
                self.black_scholes_call(
                    self.coconut_midprice_cache[-1],
                    10_000,
                    days_to_expiration,
                    0,
                    0.01011932923,
                )
            )
            # UPDATE self.bs_call_price_cache
            self.bs_call_price_cache = cache_hits[-CACHE_LIMIT:]

    def update_arb_basket_cache(self, previous_trading_state, state):
        cache_hits = previous_trading_state.get("arb_basket_cache", [])

        if self.gift_basket_midprice_cache:
            cache_hits.append(
                4 * self.chocolate_midprice_cache[-1]
                + 6 * self.strawberries_midprice_cache[-1]
                + (self.roses_midprice_cache[-1])
            )
            self.arb_basket_cache = cache_hits[-CACHE_LIMIT:]

    def update_vlad_pos(self, previous_trading_state, state):
        prev_vlad_pos = previous_trading_state.get("vlad_pos", 0)
        self.vlad_pos = prev_vlad_pos + self.get_traders_orders(
            state, "Vladimir", "GIFT_BASKET"
        )

    def get_sorted_orders(self, order_depths, product_name):
        orders = order_depths.get(product_name, OrderDepth)
        sorted_sell_orders = sorted(
            list(orders.sell_orders.items()), key=lambda x: x[0]
        )
        sorted_buy_orders = sorted(
            list(orders.buy_orders.items()), key=lambda x: x[0], reverse=True
        )
        return sorted_sell_orders, sorted_buy_orders

    def vwap(self, orders: dict) -> float:
        total_volume = sum(orders.values())
        if total_volume == 0:
            return 0
        try:
            return (
                sum(price * volume for price, volume in orders.items()) / total_volume
            )
        except:
            return 0

    def compute_z_score(self, data):
        if len(data) < 2:  # Ensure there are at least 3 data points
            return 0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0
        z_score = (data[-1] - mean) / std
        return float(z_score)

    def compute_qnorm(self, arr):
        try:
            array = np.array(arr)
            sorted_array = np.sort(array)
            norm = np.random.normal(loc=0, scale=1, size=len(array))
            norm.sort()
            qnormed_arr = np.copy(sorted_array)
            qnormed_arr[np.argsort(array)] = norm
            return qnormed_arr
        except:
            pass

    def black_scholes_call(self, S, K, T, r, sigma):
        try:
            N = statistics.NormalDist(mu=0, sigma=1)
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            call_price = S * N.cdf(d1) - K * np.exp(-r * T) * N.cdf(d2)
            return float(call_price)
        except:
            pass

    def compute_coconut_option_calcs(self):
        try:
            if len(self.coconut_coupon_midprice_cache) < 2:
                return 0
            coconut_coupon_rolling_mean = statistics.fmean(
                self.coconut_coupon_midprice_cache[-200:]
            )
            coconut_coupon_rolling_std = statistics.stdev(
                self.coconut_coupon_midprice_cache[-200:]
            )

            coconut_coupon_bsm_rolling_mean = statistics.fmean(
                self.bs_call_price_cache[-200:]
            )
            coconut_coupon_bsm_rolling_std = statistics.stdev(
                self.bs_call_price_cache[-200:]
            )

            if coconut_coupon_rolling_std != 0:
                coconut_coupon_z_score = (
                    self.coconut_coupon_midprice_cache[-1] - coconut_coupon_rolling_mean
                ) / coconut_coupon_rolling_std
            else:
                coconut_coupon_z_score = 0

            if coconut_coupon_bsm_rolling_std != 0:
                coconut_coupon_bsm_z_score = (
                    self.bs_call_price_cache[-1] - coconut_coupon_bsm_rolling_mean
                ) / coconut_coupon_bsm_rolling_std
            else:
                coconut_coupon_bsm_z_score = 0

            self.coconut_coupon_bsm_diff_z_score = (
                coconut_coupon_z_score - coconut_coupon_bsm_z_score
            )

            coconut_divergence = []
            for coconut, coupon in zip(
                self.coconut_midprice_cache, self.coconut_coupon_midprice_cache
            ):
                numerator = 10000 - coconut
                denominator = 637.63 - coupon
                if denominator != 0 and numerator != 0:
                    res = numerator / denominator
                    coconut_divergence.append(res)
                else:
                    coconut_divergence.append(0)

            self.coconut_divergence_zscore = self.compute_z_score(coconut_divergence)

        except:
            pass

    def update_gift_basket_cache(self):
        try:
            # GIFT BASKET
            gift_basket_premiums = [
                gb - bundle
                for gb, bundle in zip(
                    self.gift_basket_midprice_cache, self.arb_basket_cache
                )
            ]

            self.gift_basket_q_zscore = self.compute_z_score(
                self.compute_qnorm(gift_basket_premiums)
            )

        except:
            pass

    def calculate_acceptable_price(self, product) -> int:
        try:
            if product == "AMETHYSTS":
                return AMETHYSTS_MEAN

            if product == "STARFRUIT":
                if len(self.starfruit_vwap_cache) >= len(STARFRUIT_COEFFICIENTS) - 1:
                    expected_price = STARFRUIT_COEFFICIENTS[0] + sum(
                        STARFRUIT_COEFFICIENTS[i + 1] * self.starfruit_vwap_cache[i]
                        for i in range(len(STARFRUIT_COEFFICIENTS) - 1)
                    )
                    return int(expected_price)
                else:
                    return 0  # Not enough data to calculate price

            if product == "ORCHIDS":
                if len(self.orchids_vwap_cache) >= 1:
                    return int(self.orchids_vwap_cache[-1])
        except:
            return 0

    def generate_starfruit_and_amethysts_orders(
        self,
        state: TradingState,
        product: str,
        acceptable_price: int,
    ) -> List[Order]:
        """
        OUTPUTS:
        List[Order]: A list of Order objects, where each Order is initialized with the product name,
        the price, and the modified volume (taking into account the fudge factor).
        """
        orders: List[Order] = []
        order_depth = state.order_depths.get(product, OrderDepth())
        position = state.position.get(product, 0)
        sorted_sell_orders = sorted(
            list(order_depth.sell_orders.items()), key=lambda x: x[0]
        )

        sorted_buy_orders = sorted(
            list(order_depth.buy_orders.items()),
            key=lambda x: x[0],
            reverse=True,
        )

        best_bid = sorted_buy_orders[0][0]
        best_ask = sorted_sell_orders[0][0]

        mid_price_floor = math.floor(acceptable_price)
        mid_price_ceil = math.ceil(acceptable_price)

        position_limit = POSITION_LIMITS.get(product)
        buy_pos = position

        HALF_LIMIT = POSITION_LIMITS.get(product) // 2
        # HALF_LIMIT = 10

        # BUYING
        for ask, volume in sorted_sell_orders:
            if buy_pos >= position_limit:
                break
            if (
                ask < acceptable_price
            ):  # if someone wants to sell for less than a fair price we buy
                buy_quantity = min(abs(volume), position_limit - buy_pos)
                buy_pos += buy_quantity
                orders.append(Order(product, ask, buy_quantity))

            # if were still short we can settle for a suboptimal price to netralize
            if ask == mid_price_floor and buy_pos < 0:
                buy_quantity = min(abs(volume), -buy_pos)
                buy_pos += buy_quantity
                orders.append(Order(product, ask, buy_quantity))

        if buy_pos < position_limit:  # maximize market exposure
            if buy_pos < 0:
                s1, s2 = 0, 0
                # try to get back to neutral at a good price
                target = min(mid_price_floor + s1, best_bid + s2)
                buy_quantity = abs(buy_pos)
                buy_pos += buy_quantity
                orders.append(Order(product, target, buy_quantity))  # limit buy
            if 0 <= buy_pos and buy_pos <= HALF_LIMIT:
                s1, s2 = -2, 1
                target = min(mid_price_floor + s1, best_bid + s2)
                buy_quantity = -buy_pos + HALF_LIMIT  # get holding up to half shares
                buy_pos += buy_quantity
                orders.append(Order(product, target, buy_quantity))  # limit buy
            if buy_pos >= HALF_LIMIT:
                s1, s2 = -1, 1
                target = min(mid_price_floor + s1, best_bid + s2)
                buy_quantity = position_limit - buy_pos
                buy_pos += buy_quantity
                orders.append(Order(product, target, buy_quantity))  # limit buy

        sell_pos = position  # NOTE: set sell_pos to state.position

        # SELLING
        for bid, volume in sorted_buy_orders:
            if sell_pos <= -position_limit:
                break
            if bid > acceptable_price:
                sell_quantity = max(-(abs(volume)), -position_limit - sell_pos)
                orders.append(Order(product, bid, sell_quantity))
                sell_pos += sell_quantity
            if bid == mid_price_ceil and sell_pos > 0:  # overleveraged long
                sell_quantity = max(-volume, -sell_pos)
                sell_pos += sell_quantity
                orders.append(Order(product, bid, sell_quantity))

        if sell_pos > -position_limit:  # room to sell more
            if sell_pos > 0:  # we are long
                s1, s2 = 0, 0
                target = max(mid_price_ceil + s1, best_ask + s2)
                sell_quantity = -sell_pos
                sell_pos += sell_quantity
                orders.append(Order(product, target, sell_quantity))
            if sell_pos <= 0 and sell_pos >= -HALF_LIMIT:  # SAFE
                s1, s2 = 2, -1
                target = max(mid_price_ceil + s1, best_ask + s2)
                sell_quantity = -sell_pos - HALF_LIMIT
                sell_pos += sell_quantity
                orders.append(Order(product, target, sell_quantity))
            if sell_pos <= -HALF_LIMIT:
                s1, s2 = 2, -2
                target = max(mid_price_ceil + s1, best_ask + s2)
                sell_quantity = -position_limit - sell_pos
                sell_pos += sell_quantity
                orders.append(Order(product, target, sell_quantity))
        return orders

    def generate_orchid_orders(self, state: TradingState) -> tuple:

        orders: List[Order] = []
        order_depth = state.order_depths.get("ORCHIDS", OrderDepth())
        observations = state.observations.conversionObservations["ORCHIDS"]

        sorted_sell_orders = sorted(
            list(order_depth.sell_orders.items()), key=lambda x: x[0]
        )

        sorted_buy_orders = sorted(
            list(order_depth.buy_orders.items()),
            key=lambda x: x[0],
            reverse=True,
        )

        # mid_price_floor = math.floor(acceptable_price)
        # mid_price_ceil = math.ceil(acceptable_price)

        sunlight = observations.sunlight
        humidity = observations.humidity

        position_limit = POSITION_LIMITS.get("ORCHIDS", 100)

        transport_fees = observations.transportFees
        import_tariff = observations.importTariff
        export_tariff = observations.exportTariff

        south_bid = observations.bidPrice
        south_ask = observations.askPrice

        buy_from_south = south_ask + import_tariff + transport_fees
        sell_to_south = south_bid - export_tariff - transport_fees

        virtual_south_bid, virtual_south_ask = (
            round(buy_from_south) - 1,
            round(buy_from_south) + 1,
        )

        north_best_ask = sorted_buy_orders[0][0]
        north_best_ask = sorted_sell_orders[0][0]

        orders = []
        current_position = 0
        for ask, volume in sorted_sell_orders:
            if current_position < position_limit:
                if ask <= virtual_south_bid:
                    buy_quantity = min(-volume, position_limit - current_position)
                    current_position += buy_quantity
                    orders.append(Order("ORCHIDS", ask, buy_quantity))

        if current_position < position_limit:
            bid_price = min(north_best_ask + 1, virtual_south_bid)
            buy_quantity = position_limit - current_position
            current_position += buy_quantity
            orders.append(Order("ORCHIDS", bid_price, buy_quantity))

        current_position = 0
        for bid, volume in sorted_buy_orders:
            if current_position > -position_limit:
                if bid >= virtual_south_ask:
                    sell_quantity = max(-volume, -position_limit - current_position)
                    current_position += sell_quantity
                    orders.append(Order("ORCHIDS", bid, sell_quantity))

        if current_position > -position_limit:
            ask_price = max(north_best_ask - 5, virtual_south_ask)
            sell_quantity = -position_limit - current_position
            current_position += sell_quantity
            orders.append(Order("ORCHIDS", ask_price, sell_quantity))

        conversion_requests = -state.position.get("ORCHIDS", 0)
        return orders, conversion_requests

    def trade_gift_baskets(self, state: TradingState):
        product_orders = {
            # "CHOCOLATE": [],
            # "STRAWBERRIES": [],
            # "ROSES": [],
            "GIFT_BASKET": [],
            "COCONUT_COUPON": [],
        }
        if not (
            self.chocolate_midprice_cache
            and self.strawberries_midprice_cache
            and self.roses_midprice_cache
            and self.gift_basket_midprice_cache
        ):
            return product_orders
        product_z_score_map = {
            "GIFT_BASKET": self.gift_basket_q_zscore,
            # "CHOCOLATE": self.chocolate_q_zscore,
            "COCONUT_COUPON": self.coconut_coupon_bsm_diff_z_score,
        }
        for product, z_score in product_z_score_map.items():
            if product not in state.listings:  # for bascktester
                return product_orders

            if product == "COCONUT_COUPON":
                aggressive_bound = 1.3
                chill_bound = 1.1
                CASE_LONG_AGGRESSIVE = z_score <= -aggressive_bound
                CASE_LONG_CHILL = -aggressive_bound < z_score <= -chill_bound
                CASE_NEUTRAL = -chill_bound < z_score < chill_bound
                CASE_SHORT_CHILL = chill_bound <= z_score < aggressive_bound
                CASE_SHORT_AGGRESSIVE = z_score >= aggressive_bound

            elif product == "GIFT_BASKET":
                gb_mp_cache = self.gift_basket_midprice_cache
                arb_mp_cache = self.arb_basket_cache
                if not (
                    len(gb_mp_cache) == len(arb_mp_cache) and len(arb_mp_cache) > 100
                ):
                    return product_orders
                gb_premium = np.array(gb_mp_cache) - np.array(arb_mp_cache)
                gb_premium_slow_ma = statistics.fmean(gb_premium[-200:])
                gb_premium_fast_ma = statistics.fmean(gb_premium[-100:])

                CASE_LONG_AGGRESSIVE = CASE_LONG_CHILL = CASE_SHORT_AGGRESSIVE = (
                    CASE_SHORT_CHILL
                ) = False
                CASE_NEUTRAL = True
                if gb_premium_fast_ma > gb_premium_slow_ma + 4:
                    if gb_premium_fast_ma > gb_premium_slow_ma + 7:
                        CASE_LONG_AGGRESSIVE = True
                    else:
                        CASE_LONG_CHILL = True
                elif gb_premium_fast_ma < gb_premium_slow_ma - 4:
                    if gb_premium_fast_ma < gb_premium_slow_ma - 7:
                        CASE_SHORT_AGGRESSIVE = True
                    else:
                        CASE_SHORT_CHILL = True

            sorted_sell_orders, sorted_buy_orders = self.get_sorted_orders(
                state.order_depths, product
            )

            # LONG ON PRODUCT
            if CASE_LONG_CHILL:
                buy_pos = state.position.get(product, 0)
                for ask, volume in sorted_sell_orders:
                    if buy_pos < POSITION_LIMITS[product]:
                        buy_quantity = min(
                            abs(volume), POSITION_LIMITS[product] - buy_pos
                        )
                        buy_pos += buy_quantity
                        product_orders[product].append(
                            Order(
                                product,
                                ask,
                                buy_quantity,
                            )
                        )
            elif CASE_LONG_AGGRESSIVE:
                worst_sell = sorted_sell_orders[-1][0]
                buy_quantity = POSITION_LIMITS[product] - state.position.get(product, 0)
                product_orders[product].append(
                    Order(
                        product,
                        worst_sell,
                        buy_quantity,
                    )
                )

            # SHORT PRODUCT
            elif CASE_SHORT_CHILL:
                sell_pos = state.position.get(product, 0)
                for bid, volume in sorted_buy_orders:
                    if sell_pos > -POSITION_LIMITS[product]:
                        sell_quantity = max(
                            -volume, -POSITION_LIMITS[product] - sell_pos
                        )
                        sell_pos += sell_quantity
                        product_orders[product].append(
                            Order(
                                product,
                                bid - (2 if CASE_SHORT_AGGRESSIVE else 0),
                                sell_quantity,
                            )
                        )
            elif CASE_SHORT_AGGRESSIVE:
                worst_buy = sorted_buy_orders[-1][0]
                sell_quantity = -POSITION_LIMITS[product] - state.position.get(
                    product, 0
                )
                product_orders[product].append(
                    Order(
                        product,
                        worst_buy,
                        sell_quantity,
                    )
                )
            else:
                # Z-SCORE IN NEUTRAL ZONE
                pass

        return product_orders

    def generate_coconut_orders(self, state: TradingState, product):
        orders: List[Order] = []
        product = "COCONUT"
        order_depth = state.order_depths.get(product, OrderDepth())
        position = state.position.get(product, 0)
        sorted_sell_orders = sorted(
            list(order_depth.sell_orders.items()), key=lambda x: x[0]
        )

        sorted_buy_orders = sorted(
            list(order_depth.buy_orders.items()),
            key=lambda x: x[0],
            reverse=True,
        )
        position_limit = POSITION_LIMITS.get(product)

        # sma = self.calculate_acceptable_price(product)
        # std = self.calculate_std(product)
        # lower_band = sma - 2 * std
        # upper_band = sma + 2 * std
        if not self.coconut_midprice_cache:
            return orders
        else:
            if self.coconut_midprice_cache[-1] > 9999:
                return orders
            if self.coconut_coupon_midprice_cache[-1] > 636:
                return orders

        if self.coconut_divergence_zscore < -0.2:
            buy_pos = position
            for ask, volume in sorted_sell_orders:
                if buy_pos >= position_limit:
                    break
                buy_quantity = min(abs(volume), position_limit - buy_pos)
                buy_pos += buy_quantity
                orders.append(Order(product, ask, buy_quantity))
        if self.coconut_divergence_zscore > 2.25:
            sell_pos = position
            for bid, volume in sorted_buy_orders:
                if sell_pos <= -position_limit:
                    break
                sell_quantity = max(-(abs(volume)), -position_limit - sell_pos)
                orders.append(Order(product, bid, sell_quantity))
                sell_pos += sell_quantity

        return orders

    def get_traders_orders(self, state: TradingState, trader_name: str, product: str):
        trade = 0
        market_trades = state.market_trades.get(product, [])
        if market_trades:
            for t in market_trades:
                if t.buyer == trader_name:
                    trade += t.quantity
                elif t.seller == trader_name:
                    trade -= t.quantity
        return trade

    def get_strawberry_orders(self, state: TradingState):
        # COPY VINNIE FOR STRAWBERRIES
        orders = []
        product = "STRAWBERRIES"
        copy_trade = self.get_traders_orders(state, "Vinnie", product)
        sorted_sell_orders, sorted_buy_orders = self.get_sorted_orders(
            state.order_depths, product
        )
        current_position = state.position.get(product, 0)
        position_limit = POSITION_LIMITS.get(product)

        if copy_trade > 0:
            worst_sell = sorted_sell_orders[-1][0]
            max_buy_qty = position_limit - current_position
            orders.append(
                Order(
                    product,
                    worst_sell,
                    min(max_buy_qty, copy_trade),
                )
            )
        elif copy_trade < 0:
            worst_buy = sorted_buy_orders[-1][0]
            max_sell_qty = -position_limit - current_position
            orders.append(
                Order(
                    product,
                    worst_buy,
                    max(copy_trade, max_sell_qty),
                )
            )
        return orders

    def get_roses_orders(self, state: TradingState):
        orders = []
        product = "ROSES"
        copy_trade = self.get_traders_orders(state, "Vinnie", product)
        sorted_sell_orders, sorted_buy_orders = self.get_sorted_orders(
            state.order_depths, product
        )
        current_position = state.position.get(product, 0)
        position_limit = POSITION_LIMITS.get(product)

        if copy_trade > 0:
            worst_sell = sorted_sell_orders[-1][0]
            max_buy_qty = position_limit - current_position
            orders.append(
                Order(
                    product,
                    worst_sell,
                    min(max_buy_qty, copy_trade),
                )
            )
        elif copy_trade < 0:
            worst_buy = sorted_buy_orders[-1][0]
            max_sell_qty = -position_limit - current_position
            orders.append(
                Order(
                    product,
                    worst_buy,
                    max(copy_trade, max_sell_qty),
                )
            )
        return orders

    def get_chocolate_orders(self, state: TradingState):
        # COPY VINNIE FOR STRAWBERRIES
        orders = []
        product = "CHOCOLATE"
        copy_trade = self.get_traders_orders(state, "Vladimir", product)
        sorted_sell_orders, sorted_buy_orders = self.get_sorted_orders(
            state.order_depths, product
        )
        current_position = state.position.get(product, 0)
        position_limit = POSITION_LIMITS.get(product)

        if copy_trade > 0:
            worst_sell = sorted_sell_orders[-1][0]
            max_buy_qty = position_limit - current_position
            orders.append(
                Order(
                    product,
                    worst_sell,
                    min(max_buy_qty, copy_trade),
                )
            )
        elif copy_trade < 0:
            worst_buy = sorted_buy_orders[-1][0]
            max_sell_qty = -position_limit - current_position
            orders.append(
                Order(
                    product,
                    worst_buy,
                    max(copy_trade, max_sell_qty),
                )
            )
        return orders

    def run(self, state: TradingState):
        # SEREALIZING AND UPDATING CACHES
        previous_trading_state = self.deserialize_trader_data(state.traderData)
        # cache midprice and vwap for tradable items
        self.update_product_price_cache_listing(previous_trading_state, state)
        self.update_black_scholes_cache(previous_trading_state, state)
        self.update_arb_basket_cache(previous_trading_state, state)
        self.update_gift_basket_cache()
        self.compute_coconut_option_calcs()

        # ---- TRADING ----
        result = {}
        conversion_requests = 0
        # for product in state.listings:
        for product, _ in state.order_depths.items():
            print(product)
            acceptable_price = self.calculate_acceptable_price(product)
            if product in ["AMETHYSTS", "STARFRUIT"]:
                orders = self.generate_starfruit_and_amethysts_orders(
                    state,
                    product,
                    acceptable_price,
                )
                result[product] = orders
            elif product == "ORCHIDS":
                orders, conversion_requests = self.generate_orchid_orders(state)
                result[product] = orders
            elif product in [
                "GIFT_BASKET",
                "COCONUT_COUPON",
            ]:
                basket_orders = self.trade_gift_baskets(state)
                for key, orders in basket_orders.items():
                    if orders:
                        result[key] = orders

            elif product == "COCONUT":
                orders = self.generate_coconut_orders(state, product)
                if orders:
                    result[product] = orders

            # elif product == "STRAWBERRIES":
            #     strawberry_orders = self.get_strawberry_orders(state)
            #     result[product] = strawberry_orders

            # elif product == "ROSES":
            #     roses_orders = self.get_roses_orders(state)
            #     result[product] = roses_orders

            # elif product == "CHOCOLATE":
            #     chocolate_orders = self.get_chocolate_orders(state)
            #     result[product] = chocolate_orders

        trader_data = {
            "starfruit_vwap_cache": self.starfruit_vwap_cache,
            "orchids_vwap_cache": self.orchids_vwap_cache,
            "chocolate_midprice_cache": self.chocolate_midprice_cache,
            "strawberries_midprice_cache": self.strawberries_midprice_cache,
            "roses_midprice_cache": self.roses_midprice_cache,
            "gift_basket_midprice_cache": self.gift_basket_midprice_cache,
            "coconut_midprice_cache": self.coconut_midprice_cache,
            "coconut_coupon_midprice_cache": self.coconut_coupon_midprice_cache,
            "arb_basket_cache": self.arb_basket_cache,
            "bs_call_price_cache": self.bs_call_price_cache,
            "coconut_coupon_bsm_diff_z_score": self.coconut_coupon_bsm_diff_z_score,
            "orders_placed": result,
            "vlad_pos": self.vlad_pos,
        }

        serialized_trader_data = self.serialize_trader_data(trader_data)

        logger.flush(state, result, conversion_requests, serialized_trader_data)
        return result, conversion_requests, serialized_trader_data
