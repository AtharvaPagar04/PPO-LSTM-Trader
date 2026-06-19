from __future__ import annotations

from typing import Iterable


SUPPORTED_ASSETS = ["btc_usdt", "eth_usdt", "sol_usdt"]

BINANCE_SYMBOLS = {
    "btc_usdt": "BTCUSDT",
    "eth_usdt": "ETHUSDT",
    "sol_usdt": "SOLUSDT",
}

_ALIASES = {
    "btcusdt": "btc_usdt",
    "btc_usdt": "btc_usdt",
    "BTCUSDT": "btc_usdt",
    "ethusdt": "eth_usdt",
    "eth_usdt": "eth_usdt",
    "ETHUSDT": "eth_usdt",
    "solusdt": "sol_usdt",
    "sol_usdt": "sol_usdt",
    "SOLUSDT": "sol_usdt",
}

_LEGACY_NAME_MAP = {
    "btc_usdt": ["btcusdt"],
    "eth_usdt": ["ethusdt"],
    "sol_usdt": ["solusdt"],
}


def normalize_asset_name(asset: str) -> str:
    if not asset:
        raise ValueError("Asset name cannot be empty.")

    cleaned = asset.strip()
    normalized = _ALIASES.get(cleaned, _ALIASES.get(cleaned.lower()))
    if normalized is None:
        raise ValueError(
            f"Unsupported asset '{asset}'. Supported assets: {', '.join(SUPPORTED_ASSETS)}"
        )
    return normalized


def asset_to_symbol(asset: str) -> str:
    return BINANCE_SYMBOLS[normalize_asset_name(asset)]


def symbol_to_asset(symbol: str) -> str:
    return normalize_asset_name(symbol)


def asset_aliases(asset: str) -> list[str]:
    canonical = normalize_asset_name(asset)
    aliases = [canonical]
    aliases.extend(_LEGACY_NAME_MAP.get(canonical, []))
    symbol = asset_to_symbol(canonical)
    aliases.extend([symbol, symbol.lower()])
    return list(dict.fromkeys(aliases))


def iter_supported_assets() -> Iterable[str]:
    return list(SUPPORTED_ASSETS)
