import pytest
from unittest.mock import patch

from src.config.feature_ablation_presets import (
    available_feature_ablation_presets,
    apply_feature_preset_to_config,
    load_feature_ablation_presets,
    resolve_feature_ablation_preset,
    validate_requested_feature_ablation_presets,
    validate_feature_preset,
)
from src.experiments.feature_ablation import run_feature_ablation_experiment
from src.features.pipeline import resolve_selected_features


def test_feature_preset_file_loads_and_full_features_exists():
    presets = load_feature_ablation_presets()
    assert "full_features" in presets
    assert presets["full_features"]["features"]


def test_unknown_feature_preset_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown feature ablation preset"):
        resolve_feature_ablation_preset("missing_preset")


def test_validate_feature_preset_catches_missing_feature_names():
    with pytest.raises(ValueError, match="Unknown selected features"):
        validate_feature_preset("bad", {"features": ["missing_feature"]})


def test_apply_feature_preset_preserves_order_and_config():
    config = {"training": {"seed": 42}, "features": {"selected": ["old"]}}
    updated = apply_feature_preset_to_config(config, "price_action_minimal")
    assert updated["training"]["seed"] == 42
    assert updated["features"]["selected"] == [
        "log_return",
        "momentum_10",
        "trend",
        "rsi",
    ]


def test_new_regime_presets_exist_and_validate():
    available = available_feature_ablation_presets()
    for preset_name in [
        "minimal_plus_regime",
        "regime_trend_v1",
        "regime_volatility_v1",
        "minimal_plus_regime_rsi",
        "cross_asset_context_v1",
        "minimal_plus_cross_asset",
        "cross_asset_regime_v1",
        "stable_cross_asset_core_v1",
        "stable_cross_asset_core_v2",
    ]:
        assert preset_name in available
        resolved = resolve_feature_ablation_preset(preset_name)
        assert resolved["features"]
        assert resolve_selected_features(resolved["features"])


def test_stable_cross_asset_core_presets_match_expected_feature_lists():
    v1 = resolve_feature_ablation_preset("stable_cross_asset_core_v1")
    v2 = resolve_feature_ablation_preset("stable_cross_asset_core_v2")
    assert v1["features"] == [
        "log_return",
        "trend",
        "eth_return_24",
        "eth_return_72",
        "market_avg_return_24",
    ]
    assert v2["features"] == [
        "log_return",
        "trend",
        "eth_return_24",
        "eth_return_72",
        "sol_return_24",
        "sol_return_72",
        "market_avg_return_24",
    ]


def test_validate_requested_presets_resolves_all_before_training():
    resolved = validate_requested_feature_ablation_presets(
        ["price_action_minimal", "minimal_plus_regime"]
    )
    assert [item["features"][0] for item in resolved] == ["log_return", "log_return"]


def test_unknown_preset_error_lists_available_presets():
    with pytest.raises(ValueError) as exc_info:
        validate_requested_feature_ablation_presets(
            ["price_action_minimal", "missing_preset"]
        )
    message = str(exc_info.value)
    assert "Available presets:" in message
    assert "minimal_plus_regime" in message


def test_feature_ablation_preflight_fails_before_training():
    fake_config = {
        "data": {"window_size": 20, "train_split": 0.8},
        "training": {"seed": 42},
    }
    with patch("src.experiments.feature_ablation.train_asset") as mock_train:
        with pytest.raises(ValueError, match="Unknown feature ablation preset: missing_preset"):
            run_feature_ablation_experiment(
                "btc_usdt",
                fake_config,
                ["price_action_minimal", "missing_preset"],
                quick=True,
            )
    mock_train.assert_not_called()
