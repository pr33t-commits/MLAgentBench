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
    if text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1:
        raise ValueError("Return exactly one complete <tool_call> block")
    content, call = text.split("<tool_call>", 1)
    body, suffix = call.split("</tool_call>", 1)
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
            value = json.loads(raw, object_pairs_hook=object_pairs, parse_constant=invalid_constant)
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


class ToolCallStop:
    """Stop at a closed call outside reasoning, before argument validation."""
    def __init__(self, tokenizer, prompt_length, tools, reasoning_started=False):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.tools = tools
        self.reasoning_started = reasoning_started

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
        if "<tool_call>" not in content:
            return False
        body = content.split("<tool_call>", 1)[1]
        # A closing marker inside a still-open literal parameter is data.
        return ("</tool_call>" in body and
                body.count("<parameter=") <= body.count("</parameter>"))
