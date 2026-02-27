import time
import copy
import requests

BASE = "http://127.0.0.1:4000/v1/chat/completions"
HEADERS = {"Authorization": "Bearer sk-litellm", "Content-Type": "application/json"}

tools = [{
  "type": "function",
  "function": {
    "name": "echo",
    "description": "Echo text",
    "parameters": {
      "type": "object",
      "properties": {"text": {"type": "string"}},
      "required": ["text"]
    }
  }
}]

def post(payload):
    r = requests.post(BASE, headers=HEADERS, json=payload, timeout=120)
    return r, (r.json() if r.headers.get("content-type","").startswith("application/json") else None)

# 1) Call that should produce a tool call (NO forced tool_choice)
payload1 = {
  "model": "kimi-k2.5",
  "messages": [{
    "role": "user",
    "content": (
      "You MUST call the echo tool once with {\"text\":\"hi\"}.\n"
      "Do not answer normally. Call the tool."
    )
  }],
  "tools": tools,
  "tool_choice": "auto",
  "temperature": 1,
  "stream": False,
  "max_tokens": 200
}

msg1 = None
for attempt in range(1, 6):
    r1, j1 = post(payload1)
    print(f"call1 attempt {attempt}: {r1.status_code} {r1.text[:200]}")
    if r1.status_code != 200:
        r1.raise_for_status()
    msg = j1["choices"][0]["message"]
    if msg.get("tool_calls"):
        msg1 = msg
        break
    time.sleep(0.5)

if not msg1:
    raise RuntimeError("Model did not emit tool_calls in 5 attempts (tool_choice=auto).")

tool_call = msg1["tool_calls"][0]
tcid = tool_call.get("id")
print("tool_call_id:", tcid)
print("reasoning_content present:", "reasoning_content" in msg1)

if not tcid:
    raise RuntimeError("No tool_call id returned; cannot build replay key.")

# If the model didn't return reasoning_content, you won't reproduce the original error.
if "reasoning_content" not in msg1:
    print("WARNING: no reasoning_content returned. You may not be in thinking mode, so replay test won't trigger.")
    # You can still proceed, but it won't prove the fix.

# 2) Build follow-up history but STRIP reasoning_content (simulate stateless client)
assistant_toolcall = copy.deepcopy(msg1)
assistant_toolcall.pop("reasoning_content", None)

history = [
  payload1["messages"][0],
  assistant_toolcall,
  {"role": "tool", "tool_call_id": tcid, "content": "hi"},
]

payload2 = {
  "model": "kimi-k2.5",
  "messages": history,
  "tools": tools,
  "tool_choice": "auto",
  "temperature": 1,
  "stream": False,
  "max_tokens": 200
}

r2, j2 = post(payload2)
print("call2:", r2.status_code, r2.text[:400])
r2.raise_for_status()
print("OK: call2 succeeded (if patch is active, it should have re-injected reasoning_content when required).")
