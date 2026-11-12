"""Qwen3.5 tool dialect, schemas and validation (no inference dependencies)."""
import inspect
import json
import re


def uses_native_tools(model):
    return model.lower().startswith("huggingface/") and "qwen3.5" in model.lower()


def tool_alias(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def commentary_text(entries):
    """Replay parsed commentary only, never raw prefixes or trailing protocols."""
    fields = ("Reflection", "Research Plan and Status", "Fact Check", "Thought")
    parts = []
    for field in fields:
        if field not in entries:
            continue
        value = entries[field].strip()
        if re.search(r"<\|im_|</?(?:think|tool_call|tool_response)>|(?m:^[ \t]*Action(?: Input)?:)", value):
            value = "[Historical field omitted because it contains protocol artifacts.]"
        parts.append(f"{field}: {value}")
    return "\n\n".join(parts)


def tool_definitions(names, infos):
    tools = []
    aliases = set()
    types = {"datasets": "object", "use_demand_engine": "boolean",
             "start_line_number": "integer", "end_line_number": "integer", "offset": "integer"}
    for name in dict.fromkeys(names):
        info = infos[name]
        alias = tool_alias(name)
        if alias in aliases:
            raise ValueError(f"Duplicate tool alias: {alias}")
        aliases.add(alias)
        function = getattr(info, "function", None)
        parameters = inspect.signature(function).parameters if callable(function) else {}
        properties, required = {}, []
        for key, description in info.usage.items():
            spec = {"type": types.get(key, "string"), "description": description}
            if key == "datasets":
                spec["additionalProperties"] = {"type": "string"}
            if key == "role":
                spec["enum"] = ["baseline", "experiment"]
            if key == "stream":
                spec["enum"] = ["stdout", "stderr"]
            parameter = parameters.get(key)
            if parameter is not None and parameter.default is not inspect.Parameter.empty:
                spec["default"] = parameter.default
            else:
                required.append(key)
            properties[key] = spec
        tools.append({"type": "function", "function": {
            "name": alias, "description": f"{name}: {info.description}",
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}})
    return tools


def assistant_after_reasoning(text):
    """Separate a leading reasoning section and terminal message marker."""
    text = text.lstrip()
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            raise ValueError("Incomplete reasoning section: expected </think>")
        reasoning, text = text[len("<think>"):end], text[end + len("</think>"):]
    else:
        reasoning = ""
    text = re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$", "", text.rstrip())
    return reasoning, text


def parse_tool_response(text, tools):
    """Parse exactly one call; Qwen's <function=name> dialect is not XML proper.

    String parameters are literal text, not JSON strings or XML entities. This
    preserves code, quotes, newlines and ampersands as the official template does.
    Non-string parameters are decoded according to the advertised JSON schema.
    """
    _, text = assistant_after_reasoning(text)
    if "<tool_call>" not in text:
        raise ValueError("Return exactly one complete <tool_call> block")
    content, call = text.split("<tool_call>", 1)
    calls = split_native_calls("<tool_call>" + call)
    if len(calls) != 1:
        raise ValueError("Return exactly one complete <tool_call> block")
    body = calls[0][len("<tool_call>"):-len("</tool_call>")]
    suffix = ""
    if suffix.strip():
        raise ValueError("No text or additional calls may follow </tool_call>")
    if re.search(r"(?m)^\s*Action(?: Input)?:", content) or any(
            tag in content for tag in ("<think>", "</think>", "<tool_response>", "<|im_")):
        raise ValueError("Use assistant commentary followed by XML only; no legacy actions or role/thinking tags")
    match = re.fullmatch(r"\s*<function=([a-zA-Z0-9_]+)>\s*(.*?)\s*</function>\s*", body, re.DOTALL)
    if not match:
        raise ValueError("Expected <function=name> with named <parameter=name> blocks")
    name, parameters = match.groups()
    definitions = {t["function"]["name"]: t["function"] for t in tools}
    if name not in definitions:
        raise ValueError(f"Unknown or disabled tool: {name}")
    schema = definitions[name]["parameters"]
    arguments, position = {}, 0
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")
    while position < len(parameters):
        if not parameters[position:].strip():
            break
        parameter = re.match(r"\s*<parameter=([a-zA-Z0-9_]+)>(.*?)</parameter>",
                             parameters[position:], re.DOTALL)
        if not parameter:
            raise ValueError("Malformed parameter block")
        key, raw = parameter.groups()
        if key in arguments or key not in schema["properties"]:
            raise ValueError(f"Duplicate or unexpected parameter: {key}")
        spec = schema["properties"][key]
        # Remove only the template's one wrapper newline, preserving string data.
        raw = re.sub(r"^\r?\n", "", raw)
        raw = re.sub(r"\r?\n$", "", raw)
        kind = spec["type"]
        if kind == "string":
            value = raw
        elif kind == "boolean":
            # Jinja renders Python bool arguments as True/False in history.
            if raw.strip().lower() not in ("true", "false"):
                raise ValueError(f"{key} must be a boolean")
            value = raw.strip().lower() == "true"
        else:
            try:
                value = json.loads(raw, object_pairs_hook=object_pairs, parse_constant=invalid_constant)
            except ValueError as exc:
                raise ValueError(f"{key}: {exc}") from exc
            if kind == "integer" and type(value) is not int:
                raise ValueError(f"{key} must be an integer")
            if kind == "object" and (not isinstance(value, dict) or
                                     not all(isinstance(v, str) for v in value.values())):
                raise ValueError(f"{key} must map aliases to filename strings")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"{key} must be one of {spec['enum']}")
        arguments[key] = value
        position += parameter.end()
    missing = set(schema["required"]) - arguments.keys()
    if missing:
        raise ValueError(f"Missing required parameters: {sorted(missing)}")
    for key, spec in schema["properties"].items():
        if key not in arguments and "default" in spec:
            arguments[key] = spec["default"]
    if name == "final_answer" and not arguments.get("final_answer", "").strip():
        raise ValueError("Final Answer must contain a nonempty completed answer")
    return content.strip(), name, arguments


STATUS_FIELDS = ("reflection", "research_plan_and_status", "fact_check", "thought")


def status_tool_definition():
    return {"type": "function", "function": {"name": "record_research_status",
        "description": "Record research status; contents are not automatically fact checked.",
        "parameters": {"type": "object", "properties": {
            key: {"type": "string", "description": key.replace("_", " ")}
            for key in STATUS_FIELDS}, "required": list(STATUS_FIELDS), "additionalProperties": False}}}


def split_native_calls(text):
    """Find call boundaries while treating parameter bodies as literal text."""
    calls = []
    pos = 0
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text):
            return calls
        if not text.startswith("<tool_call>", pos):
            raise ValueError(f"Call {len(calls) + 1}: expected <tool_call>; unexpected or trailing content")
        start = pos
        pos += len("<tool_call>")
        while True:
            marker = re.search(r"<parameter=[a-zA-Z0-9_]+>|</tool_call>|<tool_call>", text[pos:])
            if not marker:
                raise ValueError(f"Call {len(calls) + 1}: incomplete tool call")
            pos += marker.start()
            token = marker.group()
            if token.startswith("<parameter="):
                end = text.find("</parameter>", pos + len(token))
                if end < 0:
                    raise ValueError(f"Call {len(calls) + 1}: incomplete {token} parameter")
                pos = end + len("</parameter>")
            elif token == "<tool_call>":
                raise ValueError(f"Call {len(calls) + 1}: nested or incomplete tool call")
            else:
                pos += len(token)
                calls.append(text[start:pos])
                break


def parse_research_response(text, tools):
    """Validate both calls completely before the caller executes any work."""
    reasoning, content = assistant_after_reasoning(text)
    calls = split_native_calls(content)
    if len(calls) != 2:
        raise ValueError(f"Expected exactly two calls: status then work; found {len(calls)}")
    parsed = []
    for index, call in enumerate(calls):
        try:
            # Protect literal tool delimiters within parameter values from the
            # single-call parser's global delimiter count.
            match = re.fullmatch(r"<tool_call>\s*<function=([a-zA-Z0-9_]+)>.*</function>\s*</tool_call>", call, re.DOTALL)
            if not match:
                raise ValueError("Malformed function or parameter blocks")
            name = match.group(1)
            if index == 0 and name != "record_research_status":
                raise ValueError(f"expected record_research_status, got {name}")
            if index == 1 and name == "record_research_status":
                raise ValueError("record_research_status cannot be the work tool")
            # Existing parser handles parameter schema/type validation.
            _, name, arguments = parse_tool_response(call, tools)
            if index == 0:
                for key in STATUS_FIELDS:
                    if not arguments[key].strip():
                        raise ValueError(f"{key} must be nonempty")
            parsed.append((name, arguments))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Call {index + 1} ({'status' if index == 0 else 'work'}): {exc}") from exc
    return reasoning, parsed[0][1], parsed[1][0], parsed[1][1]


class ToolCallStop:
    """Stop at a closed call outside reasoning, before argument validation."""
    def __init__(self, tokenizer, prompt_length, tools, reasoning_started=False, two_calls=False):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.tools = tools
        self.reasoning_started = reasoning_started
        self.two_calls = two_calls

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0][self.prompt_length:]
        # The closing delimiter is short; do not repeatedly decode the full answer.
        if not self.tokenizer.decode(generated[-32:], skip_special_tokens=False).rstrip().endswith("</tool_call>"):
            return False
        try:
            text = self.tokenizer.decode(generated, skip_special_tokens=False)
            if self.reasoning_started:
                text = "<think>" + text
            _, content = assistant_after_reasoning(text)
        except (ValueError, TypeError):
            return False
        if self.two_calls:
            try:
                return len(split_native_calls(content)) >= 2
            except ValueError:
                return False
        if "<tool_call>" not in content:
            return False
        body = content.split("<tool_call>", 1)[1]
        # A closing marker inside a still-open literal parameter is data.
        return ("</tool_call>" in body and
                body.count("<parameter=") <= body.count("</parameter>"))
