from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gr00t.model.action_head.deas_critic import CriticConfig, DEASCriticConfig
from gr00t.model.gr00t_n1_deas_critic import GR00T_N1_5, GR00T_N1_5_DEAS_Critic


def test_from_gr00t_preserves_custom_critic_config(monkeypatch):
    """Check conversion of BC config without loading weights or constructing a model."""
    requested = {"hidden_dim": 37, "depth": 2, "output_dim": 7}
    value_config = {"hidden_dim": 23, "depth": 3, "output_dim": 1}
    rl_config = {"critic_action_horizon": 16}
    pretrained = SimpleNamespace(
        config=SimpleNamespace(to_dict=lambda: {"action_head_cfg": {"hidden_size": 1024}}),
        local_model_path="/mock-pretrained",
    )
    loader = Mock(return_value=pretrained)
    monkeypatch.setattr(GR00T_N1_5, "from_pretrained", loader)

    class CapturedConfig(Exception):
        pass

    def capture_model_config(config, local_model_path):
        assert local_model_path == pretrained.local_model_path
        raise CapturedConfig(config)

    with pytest.raises(CapturedConfig) as captured:
        GR00T_N1_5_DEAS_Critic.from_pretrained.__func__(
            capture_model_config,
            "/mock-pretrained",
            from_gr00t_n1_5=True,
            critic_cfg=requested,
            value_cfg=value_config,
            rl_cfg=rl_config,
        )

    converted = captured.value.args[0].critic_cfg
    assert converted["critic_config"] == requested
    effective = CriticConfig(**converted["critic_config"])
    assert {key: getattr(effective, key) for key in requested} == requested
    assert converted["value_config"] == value_config
    assert converted["rl_config"] == rl_config
    assert converted["hidden_size"] == 1024
    assert converted["online_q_feature_passes"] == 1
    assert requested == {"hidden_dim": 37, "depth": 2, "output_dim": 7}
    loader.assert_called_once_with("/mock-pretrained")


def test_unmarked_critic_retains_legacy_feature_path():
    assert DEASCriticConfig().online_q_feature_passes == 2


@pytest.mark.parametrize("passes", [1, 2])
def test_critic_feature_marker_survives_config_serialization(passes):
    import json
    config = DEASCriticConfig(online_q_feature_passes=passes)
    restored = DEASCriticConfig(**json.loads(config.to_json_string(use_diff=False)))
    assert restored.online_q_feature_passes == passes
