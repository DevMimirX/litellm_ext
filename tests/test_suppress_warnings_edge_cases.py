def test_blank_model_cost_map_url_forces_local(monkeypatch):
    import litellm_ext.extensions.suppress_warnings as sw

    monkeypatch.setenv("LITELLM_MODEL_COST_MAP_URL", "")
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)

    sw._ensure_local_model_cost_map_for_blank_url()

    assert sw.os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "true"


def test_non_blank_invalid_model_cost_map_url_does_not_force_local(monkeypatch):
    import litellm_ext.extensions.suppress_warnings as sw

    monkeypatch.setenv("LITELLM_MODEL_COST_MAP_URL", "htp://bad")
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)

    sw._ensure_local_model_cost_map_for_blank_url()

    assert sw.os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") is None


def test_warning_filter_matches_blank_url_only():
    import litellm_ext.extensions.suppress_warnings as sw

    msg = "LiteLLM: Failed to fetch remote model cost map from %s: %s. Falling back to local backup."
    err = "Request URL is missing an 'http://' or 'https://' protocol."

    assert sw._is_model_cost_map_url_warning(msg, ("", err)) is True
    assert sw._is_model_cost_map_url_warning(msg, ("htp://bad", err)) is False
