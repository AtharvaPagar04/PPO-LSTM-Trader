import pytest

from src.config.assets import asset_to_symbol, normalize_asset_name


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("btcusdt", "btc_usdt"),
        ("BTCUSDT", "btc_usdt"),
        ("btc_usdt", "btc_usdt"),
        ("ethusdt", "eth_usdt"),
        ("eth_usdt", "eth_usdt"),
        ("solusdt", "sol_usdt"),
        ("sol_usdt", "sol_usdt"),
    ],
)
def test_normalize_asset_name(raw_name, expected):
    assert normalize_asset_name(raw_name) == expected


def test_unsupported_asset_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported asset"):
        normalize_asset_name("doge_usdt")


def test_asset_to_symbol_uses_canonical_names():
    assert asset_to_symbol("ethusdt") == "ETHUSDT"
