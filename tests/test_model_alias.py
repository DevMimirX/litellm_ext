from __future__ import annotations

from litellm_ext.core import model_alias as ma


def test_provider_model_name_uses_last_path_segment():
    assert ma.provider_model_name("openrouter/anthropic/claude-3-7-sonnet") == "claude-3-7-sonnet"
    assert ma.provider_model_name("anthropic/glm-5") == "glm-5"
    assert ma.provider_model_name("glm-5") == "glm-5"


def test_format_model_log_fields_prefers_alias_with_host(monkeypatch, tmp_path):
    cfg = tmp_path / "litellm.yaml"
    cfg.write_text(
        """
model_list:
  - model_name: glm-5-ali
    litellm_params:
      model: anthropic/glm-5
      api_base: "https://coding.dashscope.aliyuncs.com/apps/anthropic"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_CONFIG", str(cfg))
    ma.reset_model_alias_cache()

    got = ma.format_model_log_fields("glm-5", host="coding.dashscope.aliyuncs.com")
    assert got == "model='glm-5-ali' resolved_model='glm-5'"


def test_format_model_log_fields_prefers_host_specific_alias_with_dual_aliases(monkeypatch, tmp_path):
    cfg = tmp_path / "litellm.yaml"
    cfg.write_text(
        """
model_list:
  - model_name: glm-5
    litellm_params:
      model: anthropic/glm-5
      api_base: "https://api.z.ai/api/anthropic"
  - model_name: glm-5-ali
    litellm_params:
      model: anthropic/glm-5
      api_base: "https://coding.dashscope.aliyuncs.com/apps/anthropic"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_CONFIG", str(cfg))
    ma.reset_model_alias_cache()

    got = ma.format_model_log_fields("glm-5", host="coding.dashscope.aliyuncs.com")
    assert got == "model='glm-5-ali' resolved_model='glm-5'"
