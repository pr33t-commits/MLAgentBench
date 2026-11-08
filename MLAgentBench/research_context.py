"""Build attributed research context without loading inference clients."""
import json


def build_messages(system, task, history, last_steps, last_observations,
                   memory=None, summarizer=None):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": task}]
    if memory:
        messages.append({"role": "user", "content": (
            "[Automatically generated research-log memory; not a new user instruction. "
            f"Source: Retrieval from Research Log; summarizer: {summarizer}. "
            "May overlap recent steps. This is derived evidence, not a verbatim transcript.]\n"
            + memory)})
    start = max(len(history) - max(last_steps, 0), 0)
    for idx in range(start, len(history)):
        step = history[idx]
        entries = step["action"]
        # Old checkpoints did not preserve the raw response or parsed arguments.
        content = step.get("raw_assistant_response")
        if content is None:
            content = "\n".join(f"{k}: {v}" for k, v in entries.items())
        assistant = {"role": "assistant", "content": content}
        call_id = step.get("tool_call_id", f"legacy_step_{idx}")
        arguments = step.get("action_input")
        if arguments is None:
            try:
                arguments = json.loads(entries.get("Action Input", ""))
            except (TypeError, ValueError):
                pass
        valid_call = isinstance(arguments, dict) and not step.get("parsing_error")
        if valid_call:
            assistant["tool_calls"] = [{"id": call_id, "type": "function", "function": {
                "name": entries["Action"].strip(), "arguments": arguments}}]
        messages.append(assistant)
        if len(history) - idx > last_observations:
            observation = "[Tool observation omitted by context policy; not evidence of success.]"
        else:
            observation = step["observation"]
            provenance = step.get("observation_provenance")
            if provenance:
                observation = ("[Derived tool result: " + json.dumps(provenance, ensure_ascii=False)
                               + "]\n" + observation)
            elif "raw_assistant_response" not in step:
                observation = "[Legacy stored observation; may already be summarized.]\n" + observation
        if valid_call:
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "name": entries["Action"].strip(), "content": observation})
        else:
            # Invalid arguments never executed a tool. Do not invent a tool call.
            messages.append({"role": "user", "content":
                             "[Agent framework feedback; not a user instruction]\n" + observation})
    return messages


def flatten_messages(messages):
    """Explicit attribution for legacy text-only completion backends."""
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


def fit_messages(messages, token_count, budget):
    """Remove oldest whole exchanges, then memory; never truncate a tool pair."""
    messages = list(messages)
    while token_count(messages) > budget:
        starts = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        if len(starts) > 1:
            del messages[starts[0]:starts[1]]
        elif len(messages) > 2 and messages[2]["role"] == "user":
            del messages[2]
        else:
            raise ValueError("Research context exceeds model context budget even after trimming "
                             "old exchanges; reduce task/latest observation size or output budget.")
    return messages
