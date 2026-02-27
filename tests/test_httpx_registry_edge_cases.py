from __future__ import annotations

import httpx


def test_install_idempotent_send_wrapped_once():
    import litellm_ext.core.registry as reg

    reg.install_httpx_patch()
    first = httpx.Client.send
    reg.install_httpx_patch()
    second = httpx.Client.send

    assert first is second, "install_httpx_patch() must be idempotent (no nested wrappers)"


def test_mutator_priority_replacement_and_exception_does_not_break_chain():
    import litellm_ext.core.registry as reg

    reg.install_httpx_patch()

    calls = []

    def bad_mutator(_req: httpx.Request):
        calls.append("bad")
        raise RuntimeError("boom")

    def none_mutator(_req: httpx.Request):
        calls.append("none")
        return None

    def stub_mutator(_req: httpx.Request):
        calls.append("stub")
        return httpx.Response(200, json={"stub": True}, request=_req)

    # Ensure ordering: lower priority runs first
    reg.register_request_mutator("bad", bad_mutator, priority=1)
    reg.register_request_mutator("none", none_mutator, priority=5)
    reg.register_request_mutator("stub", stub_mutator, priority=10)

    # Pass-through transport should never be hit (stub returns first)
    transport = httpx.MockTransport(lambda req: httpx.Response(418, json={"hit_transport": True}, request=req))
    with httpx.Client(transport=transport) as c:
        r = c.post("https://example.com/anything", json={"x": 1})

    assert r.status_code == 200
    assert r.json() == {"stub": True}
    assert calls == ["bad", "none", "stub"]

    # Replacement: same name must replace
    def stub2(_req: httpx.Request):
        return httpx.Response(200, json={"stub": 2}, request=_req)

    reg.register_request_mutator("stub", stub2, priority=10)

    with httpx.Client(transport=transport) as c:
        r2 = c.post("https://example.com/anything", json={"x": 1})

    assert r2.json() == {"stub": 2}


def test_async_send_uses_mutators_and_can_short_circuit():
    import litellm_ext.core.registry as reg

    reg.install_httpx_patch()

    def stub(_req: httpx.Request):
        return httpx.Response(200, json={"async": True}, request=_req)

    reg.register_request_mutator("stub", stub, priority=1)

    async def _run():
        transport = httpx.MockTransport(lambda req: httpx.Response(418, json={"hit_transport": True}, request=req))
        async with httpx.AsyncClient(transport=transport) as c:
            return await c.post("https://example.com/anything", json={"x": 1})

    import asyncio
    resp = asyncio.run(_run())
    assert resp.status_code == 200
    assert resp.json() == {"async": True}
