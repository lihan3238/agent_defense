from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from time import perf_counter
from typing import Any, Literal

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)

from agent_defense.artifacts import DetectorArtifact

_FUNCTION_PATTERN = re.compile(r"<function\s*=\s*([^>]+)>(.*?)</function>", re.DOTALL)
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_EMPTY_ENV = EmptyEnv()
_CAPTURE_POSITION_NAMES = {
    "tool_input": "generation_prefill_last_nonpad",
    "function_call": "function_call_end",
}
_PREFLIGHT_COMPATIBILITY_KEYS = (
    "revision",
    "model_dtype",
    "quantization_config_hash",
    "tokenizer_class",
    "state_kind",
    "module_path",
    "chat_template_hash",
)
_LOCAL_CHECKPOINT_ID_VERSION = "local-checkpoint-v1"
_LOCAL_CHECKPOINT_METADATA_NAMES = {
    "adapter_config.json",
    "awq_config.json",
    "config.json",
    "generation_config.json",
    "gptq_config.json",
    "model.safetensors.index.json",
    "params.json",
    "preprocessor_config.json",
    "processor_config.json",
    "pytorch_model.bin.index.json",
    "quantization_config.json",
    "quantize_config.json",
    "tokenizer_config.json",
}
_LOCAL_CHECKPOINT_METADATA_PREFIXES = (
    "added_tokens",
    "chat_template",
    "merges",
    "sentencepiece",
    "special_tokens_map",
    "spiece",
    "tekken",
    "tokenizer",
    "vocab",
)
_LOCAL_CHECKPOINT_WEIGHT_INDEX_NAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_LOCAL_CHECKPOINT_SINGLE_WEIGHT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "adapter_model.safetensors",
    "adapter_model.bin",
)
_LOCAL_CHECKPOINT_SAMPLE_BYTES = 64 * 1024


@dataclass(frozen=True)
class _ToolCallParseResult:
    calls: list[FunctionCall]
    errors: int
    spans: list[tuple[int, int]]


def _parse_tool_call_details(
    text: str,
    *,
    call_id: str | None = None,
) -> _ToolCallParseResult:
    """Parse complete non-overlapping tags and count only actual malformed candidates."""

    tagged_matches = [
        *((match.start(), match.end(), "function", match) for match in _FUNCTION_PATTERN.finditer(text)),
        *((match.start(), match.end(), "tool_call", match) for match in _TOOL_CALL_PATTERN.finditer(text)),
    ]
    tagged_matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    calls: list[FunctionCall] = []
    valid_spans: list[tuple[int, int]] = []
    errors = 0
    covered_until = -1
    covered_spans: list[tuple[int, int]] = []
    for start, end, kind, match in tagged_matches:
        if start < covered_until:
            continue
        covered_until = end
        covered_spans.append((start, end))
        if kind == "function":
            function_name = match.group(1).strip()
            raw_payload = match.group(2).strip()
            try:
                args = json.loads(raw_payload)
            except json.JSONDecodeError:
                errors += 1
                continue
            if not function_name or not isinstance(args, dict):
                errors += 1
                continue
        else:
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                errors += 1
                continue
            if not isinstance(payload, dict):
                errors += 1
                continue
            function_name = payload.get("name") or payload.get("function")
            args = payload.get("arguments") or payload.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    errors += 1
                    continue
            if not isinstance(function_name, str) or not function_name or not isinstance(args, dict):
                errors += 1
                continue
        index = len(calls)
        item_id = call_id if index == 0 else f"{call_id}-{index}" if call_id else None
        calls.append(FunctionCall(function=function_name, args=args, id=item_id))
        valid_spans.append((start, end))

    residual = list(text)
    for start, end in covered_spans:
        residual[start:end] = " " * (end - start)
    residual_text = "".join(residual)
    errors += len(re.findall(r"<function\s*=", residual_text))
    errors += residual_text.count("<tool_call>")
    return _ToolCallParseResult(calls=calls, errors=errors, spans=valid_spans)


def _parse_tool_call_with_diagnostics(
    text: str,
    *,
    call_id: str | None = None,
) -> tuple[list[FunctionCall], int]:
    result = _parse_tool_call_details(text, call_id=call_id)
    return result.calls, result.errors


def parse_tool_call(text: str, *, call_id: str | None = None) -> list[FunctionCall]:
    """Parse AgentDojo-style or Qwen-style tool calls without executing them."""

    calls, _ = _parse_tool_call_with_diagnostics(text, call_id=call_id)
    return calls


def _structured_tool_markup_count(text: str) -> int:
    """Deprecated compatibility helper returning malformed candidate count."""

    _, errors = _parse_tool_call_with_diagnostics(text)
    return errors


def _tool_schemas(tools: Collection[Function]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.model_json_schema(),
            },
        }
        for tool in tools
    ]


def _chat_template_hash(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str):
        source = template
    elif isinstance(template, Mapping):
        source = json.dumps(template, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        source = f"no-template:{type(tokenizer).__name__}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _tool_schema_hash(tools: Collection[Function]) -> str:
    serialized = json.dumps(_tool_schemas(tools), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _quantization_config_hash(model: Any) -> str:
    config = getattr(model.config, "quantization_config", None)
    if config is None:
        return "none"
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    serialized = json.dumps(config, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _full_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sampled_file_sha256(path: Path, size: int) -> str:
    """Hash bounded, deterministic regions without reading an entire weight shard."""

    sample_size = min(size, _LOCAL_CHECKPOINT_SAMPLE_BYTES)
    offsets = sorted(
        {
            0,
            max(0, (size - sample_size) // 2),
            max(0, size - sample_size),
        }
    )
    digest = hashlib.sha256()
    digest.update(f"size={size}\n".encode())
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            chunk = handle.read(sample_size)
            digest.update(f"offset={offset};length={len(chunk)}\n".encode())
            digest.update(chunk)
    return digest.hexdigest()


def _is_checkpoint_metadata_file(path: Path, *, include_remote_code: bool) -> bool:
    name = path.name.casefold()
    return (
        name in _LOCAL_CHECKPOINT_METADATA_NAMES
        or name.startswith(_LOCAL_CHECKPOINT_METADATA_PREFIXES)
        or (
            include_remote_code
            and path.suffix.casefold() == ".py"
            and name.startswith(
                (
                    "configuration_",
                    "image_processing_",
                    "modeling_",
                    "processing_",
                    "tokenization_",
                )
            )
        )
    )


def _indexed_weight_paths(checkpoint_path: Path, index_paths: Sequence[Path]) -> list[Path]:
    relative_names: set[str] = set()
    for index_path in index_paths:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload["weight_map"]
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Local checkpoint has an invalid weight index") from error
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError("Local checkpoint weight index must contain a non-empty weight_map")
        for shard_name in weight_map.values():
            if not isinstance(shard_name, str) or not shard_name:
                raise ValueError("Local checkpoint weight index contains an invalid shard name")
            relative = Path(shard_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Local checkpoint weight index contains an unsafe shard path")
            relative_names.add(shard_name)

    paths = [checkpoint_path / relative_name for relative_name in sorted(relative_names)]
    if any(not path.is_file() for path in paths):
        raise ValueError("Local checkpoint weight index references a missing shard")
    return paths


def _local_checkpoint_content_id(
    model_id_or_path: str,
    *,
    include_remote_code: bool = False,
) -> str | None:
    """Return a path-independent, low-cost identity for a local HF checkpoint.

    Small config/tokenizer/index files are hashed in full. Large weight files are
    represented by their name, size, and fixed first/middle/last samples, avoiding
    a full read of multi-gigabyte checkpoints while still distinguishing same-layout
    local weight variants in normal use. Remote HF repository IDs return ``None``
    and remain bound by their resolved immutable revision instead.
    """

    checkpoint_path = Path(model_id_or_path)
    if not checkpoint_path.exists():
        return None
    if checkpoint_path.is_file():
        metadata_paths = (
            [checkpoint_path]
            if _is_checkpoint_metadata_file(
                checkpoint_path,
                include_remote_code=include_remote_code,
            )
            else []
        )
        weight_paths = [] if metadata_paths else [checkpoint_path]
    elif checkpoint_path.is_dir():
        root_files = sorted(
            (path for path in checkpoint_path.iterdir() if path.is_file()),
            key=lambda path: path.name,
        )
        metadata_paths = [
            path
            for path in root_files
            if _is_checkpoint_metadata_file(path, include_remote_code=include_remote_code)
        ]
        index_paths = [
            checkpoint_path / name
            for name in _LOCAL_CHECKPOINT_WEIGHT_INDEX_NAMES
            if (checkpoint_path / name).is_file()
        ]
        weight_paths = _indexed_weight_paths(checkpoint_path, index_paths) if index_paths else []
        weight_paths.extend(
            checkpoint_path / name
            for name in _LOCAL_CHECKPOINT_SINGLE_WEIGHT_NAMES
            if (checkpoint_path / name).is_file()
        )
        weight_paths = sorted(
            set(weight_paths), key=lambda path: path.relative_to(checkpoint_path).as_posix()
        )
    else:
        raise ValueError("Local model path is neither a checkpoint file nor directory")

    entries: list[dict[str, str | int]] = []
    for path in metadata_paths:
        size = path.stat().st_size
        relative_name = (
            path.name if checkpoint_path.is_file() else path.relative_to(checkpoint_path).as_posix()
        )
        entries.append(
            {
                "kind": "metadata_full_sha256",
                "name": relative_name,
                "size": size,
                "sha256": _full_file_sha256(path),
            }
        )
    for path in weight_paths:
        size = path.stat().st_size
        relative_name = (
            path.name if checkpoint_path.is_file() else path.relative_to(checkpoint_path).as_posix()
        )
        entries.append(
            {
                "kind": "weight_bounded_sha256",
                "name": relative_name,
                "size": size,
                "sha256": _sampled_file_sha256(path, size),
            }
        )
    if not entries:
        raise ValueError("Local model path contains no recognized checkpoint identity files")

    serialized = json.dumps(
        {"schema": _LOCAL_CHECKPOINT_ID_VERSION, "files": entries},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{_LOCAL_CHECKPOINT_ID_VERSION}:{hashlib.sha256(serialized.encode()).hexdigest()}"


def _resolved_revision(model: Any, requested_revision: str | None) -> str | None:
    """Prefer the immutable commit resolved by Transformers over a mutable label."""

    return getattr(model.config, "_commit_hash", None) or requested_revision


def _display_model_id(model_id_or_path: str) -> str:
    """Keep public model IDs while removing local filesystem prefixes."""

    path = Path(model_id_or_path)
    if path.is_absolute() or path.exists():
        return path.name
    if re.match(r"^[A-Za-z]:[\\/]", model_id_or_path):
        return PureWindowsPath(model_id_or_path).name
    return model_id_or_path


def _system_message_hash(messages: Sequence[ChatMessage]) -> str:
    system_text = "\n\n".join(
        get_text_content_as_str(message.get("content") or [])
        for message in messages
        if message["role"] == "system"
    )
    return hashlib.sha256(system_text.encode("utf-8")).hexdigest()


def _tool_prompt(tools: Collection[Function]) -> str:
    if not tools:
        return ""
    schemas = [schema["function"] for schema in _tool_schemas(tools)]
    return (
        "\n\n# Available tools\n"
        + json.dumps(schemas, ensure_ascii=False, indent=2)
        + "\n\nWhen a tool is needed, emit exactly one call and stop:\n"
        + '<function=tool_name>{"argument":"value"}</function>\n'
        + "Otherwise answer normally. Never invent an unavailable tool."
    )


def _assistant_tool_text(message: ChatMessage) -> str:
    calls = message.get("tool_calls") if message["role"] == "assistant" else None
    if not calls:
        return ""
    return "\n".join(
        f"<function={call.function}>{json.dumps(dict(call.args), ensure_ascii=False)}</function>"
        for call in calls
    )


def _strip_structured_tool_markup(text: str) -> str:
    """Keep assistant prose while avoiding duplicate rendered tool calls."""

    return _TOOL_CALL_PATTERN.sub("", _FUNCTION_PATTERN.sub("", text)).strip()


def _tool_call_end_length(
    tokenizer: Any,
    generated_ids: Any,
    raw_completion: str,
    *,
    target_char: int | None = None,
) -> int:
    """Locate the closing-tag boundary using the model's original generated IDs."""

    if target_char is None:
        closing_boundaries: list[int] = []
        for marker in ("</tool_call>", "</function>"):
            index = raw_completion.rfind(marker)
            if index >= 0:
                closing_boundaries.append(index + len(marker))
        if not closing_boundaries:
            raise ValueError("Could not locate a closing tool-call tag in the generated completion")
        target_char = max(closing_boundaries)
    if not 0 < target_char <= len(raw_completion):
        raise ValueError("Tool-call character boundary is outside the generated completion")
    for length in range(1, int(generated_ids.shape[-1]) + 1):
        prefix = tokenizer.decode(generated_ids[:length], skip_special_tokens=True)
        if len(prefix) >= target_char and prefix[:target_char] == raw_completion[:target_char]:
            return length
    raise ValueError("Could not map the closing tool-call tag back to the original generated token IDs")


def render_agentdojo_messages(
    messages: Sequence[ChatMessage],
    tools: Collection[Function],
) -> list[dict[str, str]]:
    """Convert AgentDojo messages to a conservative, model-neutral chat transcript."""

    rendered: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        content = get_text_content_as_str(message.get("content") or [])
        if role == "system":
            rendered.append({"role": "system", "content": content + _tool_prompt(tools)})
        elif role == "tool":
            tool_name = message["tool_call"].function
            error = message.get("error")
            prefix = f"[TOOL RESULT: {tool_name}]"
            if error:
                content = f"{prefix} ERROR: {error}\n{content}"
            else:
                content = f"{prefix}\n{content}"
            # User role is accepted by more chat templates than an unpaired tool role.
            rendered.append({"role": "user", "content": content})
        else:
            tool_text = _assistant_tool_text(message)
            prose = _strip_structured_tool_markup(content) if tool_text else content
            combined = "\n".join(part for part in (prose, tool_text) if part)
            rendered.append({"role": role, "content": combined})
    return rendered


def render_native_agentdojo_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Render messages for chat templates with native ``tools=`` support."""

    rendered: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = get_text_content_as_str(message.get("content") or [])
        if role == "assistant" and message.get("tool_calls"):
            content = _strip_structured_tool_markup(content)
        item: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and message.get("tool_calls"):
            item["tool_calls"] = [
                {
                    "type": "function",
                    "function": {"name": call.function, "arguments": dict(call.args)},
                }
                for call in message["tool_calls"] or []
            ]
        if role == "tool":
            item["name"] = message["tool_call"].function
        rendered.append(item)
    return rendered


def _find_decoder_layers(model: Any) -> tuple[Any, str]:
    candidates = (
        ("model.layers", lambda value: value.model.layers),
        ("transformer.h", lambda value: value.transformer.h),
        ("gpt_neox.layers", lambda value: value.gpt_neox.layers),
        ("model.decoder.layers", lambda value: value.model.decoder.layers),
    )
    for path, getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if len(layers):
            return layers, path
    raise ValueError("Could not locate transformer decoder blocks for residual-stream capture")


class _ResidualPreCapture:
    def __init__(self, model: Any, layer: int, attention_mask: Any) -> None:
        self.model = model
        self.layers, self.module_path = _find_decoder_layers(model)
        self.layer = layer if layer >= 0 else len(self.layers) + layer
        if not 0 <= self.layer < len(self.layers):
            raise ValueError(f"Layer {layer} is outside a model with {len(self.layers)} blocks")
        self.attention_mask = attention_mask
        self.value: Any | None = None
        self.handle: Any | None = None

    def _hook(self, module: Any, args: tuple[Any, ...]) -> None:
        import torch

        del module
        if self.value is not None:
            return
        hidden = args[0]
        mask = self.attention_mask[:, : hidden.shape[1]].to(hidden.device)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        last_indices = (mask * positions).argmax(dim=1)
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        self.value = hidden[batch_indices, last_indices].detach().float().cpu()[0].numpy()

    def __enter__(self) -> _ResidualPreCapture:
        self.handle = self.layers[self.layer].register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        if self.handle is not None:
            self.handle.remove()


class HuggingFaceToolCallingLLM(BasePipelineElement):
    """In-process HF agent that exposes a pre-tool-call residual-stream snapshot."""

    def __init__(
        self,
        model_id_or_path: str,
        *,
        layer: int = -1,
        position: Literal["tool_input", "function_call"] = "tool_input",
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        seed: int = 0,
        disable_thinking: bool = False,
        capture_activation: bool = True,
        activation_metadata: Mapping[str, Any] | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.display_model_id = _display_model_id(model_id_or_path)
        self.layer = layer
        self.position = position
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.disable_thinking = disable_thinking
        self.capture_activation = capture_activation
        self.activation_metadata = dict(activation_metadata or {})
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.display_model_id).strip("-")
        self.name = f"local-{safe_model_name or 'model'}"
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self.query_count = 0
        self.parsed_tool_call_count = 0
        self.tool_parse_error_count = 0
        self.generate_elapsed_ms = 0.0
        self.replay_forward_count = 0
        self._checkpoint_content_id_loaded = False
        self._checkpoint_content_id_value: str | None = None

    def attach_loaded_backend(self, model: Any, tokenizer: Any) -> None:
        """Reuse one already-loaded model/tokenizer pair across sequential trials."""

        if model is None or tokenizer is None:
            raise ValueError("A shared HF backend requires both model and tokenizer")
        self._model = model
        self._tokenizer = tokenizer

    def loaded_backend(self) -> tuple[Any, Any] | None:
        """Return the in-process backend without triggering a model load."""

        if self._model is None or self._tokenizer is None:
            return None
        return self._model, self._tokenizer

    def ensure_loaded(self) -> tuple[Any, Any]:
        """Load the backend before trial timing so all defenses share one timing boundary."""

        return self._load()

    def checkpoint_content_id(self) -> str | None:
        """Return the cached local checkpoint identity, if the source is a local path."""

        if not self._checkpoint_content_id_loaded:
            self._checkpoint_content_id_value = _local_checkpoint_content_id(
                self.model_id_or_path,
                include_remote_code=self.trust_remote_code,
            )
            self._checkpoint_content_id_loaded = True
        return self._checkpoint_content_id_value

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError("Install the HF extra with `uv sync --extra hf`.") from error
        common = {
            "revision": self.revision,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path, **common)
        model_kwargs: dict[str, Any] = {**common, "torch_dtype": self.dtype}
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id_or_path, **model_kwargs)
        if self.device != "auto":
            self._model.to(torch.device(self.device))
        self._model.eval()
        return self._model, self._tokenizer

    def _render(
        self,
        tokenizer: Any,
        native_messages: list[dict[str, Any]],
        fallback_messages: list[dict[str, str]],
        tools: Collection[Function],
    ) -> tuple[str, str]:
        template_kwargs = {"enable_thinking": False} if self.disable_thinking else {}
        render_suffix = ":disable_thinking" if self.disable_thinking else ""
        if tools:
            try:
                rendered = tokenizer.apply_chat_template(
                    native_messages,
                    tools=_tool_schemas(tools),
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
                return rendered, f"native_tools{render_suffix}"
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
        try:
            rendered = tokenizer.apply_chat_template(
                fallback_messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            return rendered, f"generic_tool_prompt{render_suffix}"
        except (AttributeError, TypeError, ValueError):
            rendered = (
                "\n\n".join(
                    f"[{message['role'].upper()}]\n{message['content']}" for message in fallback_messages
                )
                + "\n\n[ASSISTANT]\n"
            )
            return rendered, f"plain_text_fallback{render_suffix}"

    def capture_text_activation(self, text: str) -> tuple[Any, dict[str, Any]]:
        """Run a standalone forward pass for an installation/shape smoke test.

        This helper does not represent an AgentDojo experiment. The runtime path in
        :meth:`query` captures the actual generation prefill instead.
        """

        model, tokenizer = self._load()
        try:
            import torch
        except ImportError as error:
            raise ImportError("Install the HF extra with `uv sync --extra hf`.") from error
        encoded = tokenizer(text, return_tensors="pt")
        model_device = next(model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        with torch.inference_mode(), _ResidualPreCapture(model, self.layer, attention_mask) as capture:
            model(**encoded, use_cache=False)
        if capture.value is None:
            raise RuntimeError("Residual-stream hook did not capture a value")
        last_index = int(attention_mask[0].nonzero()[-1])
        token_id = int(input_ids[0, last_index])
        metadata = {
            "model_id": self.display_model_id,
            "revision": _resolved_revision(model, self.revision),
            "model_dtype": str(next(model.parameters()).dtype),
            "quantization_config_hash": _quantization_config_hash(model),
            "tokenizer_class": type(tokenizer).__name__,
            "layer": capture.layer,
            "module_path": f"{capture.module_path}.{capture.layer}",
            "state_kind": "resid_pre",
            "position": "standalone_text_last_nonpad",
            "token_id": token_id,
            "token_text": tokenizer.decode([token_id]),
            "input_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "extra_forward_count": 1,
        }
        if checkpoint_content_id := self.checkpoint_content_id():
            metadata["checkpoint_content_id"] = checkpoint_content_id
        return capture.value, metadata

    def preflight_artifact(self, artifact: DetectorArtifact) -> dict[str, Any]:
        """Reject model/layer/width identity mismatches before an AgentDojo trial."""

        model, tokenizer = self._load()
        layers, module_path = _find_decoder_layers(model)
        resolved_layer = self.layer if self.layer >= 0 else len(layers) + self.layer
        if not 0 <= resolved_layer < len(layers):
            raise ValueError(f"Layer {self.layer} is outside a model with {len(layers)} blocks")
        expected_position = _CAPTURE_POSITION_NAMES[self.position]
        hidden_size = getattr(model.config, "hidden_size", None)
        if artifact.model_id != self.display_model_id:
            raise ValueError(
                f"Artifact model_id={artifact.model_id!r} does not match model {self.display_model_id!r}"
            )
        if artifact.layer != resolved_layer:
            raise ValueError(
                f"Artifact layer={artifact.layer} does not match resolved model layer={resolved_layer}"
            )
        if artifact.position != expected_position:
            raise ValueError(
                f"Artifact position={artifact.position!r} does not match runtime {expected_position!r}"
            )
        if hidden_size is not None and artifact.dimension != int(hidden_size):
            raise ValueError(
                f"Artifact dimension={artifact.dimension} does not match model hidden_size={hidden_size}"
            )
        actual = {
            "revision": _resolved_revision(model, self.revision),
            "model_dtype": str(next(model.parameters()).dtype),
            "quantization_config_hash": _quantization_config_hash(model),
            "tokenizer_class": type(tokenizer).__name__,
            "state_kind": "resid_pre",
            "module_path": f"{module_path}.{resolved_layer}",
            "chat_template_hash": _chat_template_hash(tokenizer),
        }
        checkpoint_content_id = self.checkpoint_content_id()
        if checkpoint_content_id is not None:
            actual["checkpoint_content_id"] = checkpoint_content_id
        compatibility = artifact.metadata.get("activation_compatibility", {})
        if not isinstance(compatibility, Mapping):
            raise ValueError("Artifact activation_compatibility must be a mapping")
        required_keys = (
            *_PREFLIGHT_COMPATIBILITY_KEYS,
            *(("checkpoint_content_id",) if checkpoint_content_id is not None else ()),
        )
        missing = [key for key in required_keys if key not in compatibility]
        if missing:
            raise ValueError(f"Artifact is missing preflight compatibility keys: {missing}")
        if checkpoint_content_id is None and "checkpoint_content_id" in compatibility:
            raise ValueError("Artifact checkpoint_content_id requires the matching local checkpoint source")
        for key, actual_value in actual.items():
            if key in compatibility and compatibility[key] != actual_value:
                raise ValueError(
                    f"Artifact {key}={compatibility[key]!r} does not match runtime {actual_value!r}"
                )
        return {
            "model_id": self.display_model_id,
            "layer": resolved_layer,
            "position": expected_position,
            "dimension": int(hidden_size) if hidden_size is not None else artifact.dimension,
            **actual,
        }

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = _EMPTY_ENV,
        messages: Sequence[ChatMessage] = (),
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict[str, Any]]:
        state = {} if extra_args is None else dict(extra_args)
        self.query_count += 1
        for key in (
            "agent_defense.activation",
            "agent_defense.activation_metadata",
            "agent_defense.activation_error",
        ):
            state.pop(key, None)
        model, tokenizer = self._load()
        try:
            import torch
        except ImportError as error:
            raise ImportError("Install the HF extra with `uv sync --extra hf`.") from error
        tools = tuple(runtime.functions.values())
        fallback_chat = render_agentdojo_messages(messages, tools)
        native_chat = render_native_agentdojo_messages(messages)
        rendered, render_mode = self._render(tokenizer, native_chat, fallback_chat, tools)
        encoded = tokenizer(rendered, return_tensors="pt")
        model_device = next(model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        capture: _ResidualPreCapture | None = None

        generate_started = perf_counter()
        try:
            with torch.inference_mode():
                torch.manual_seed(self.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)
                if self.capture_activation and self.position == "tool_input":
                    with _ResidualPreCapture(model, self.layer, attention_mask) as capture:
                        output_ids = model.generate(
                            **encoded,
                            do_sample=False,
                            max_new_tokens=self.max_new_tokens,
                            pad_token_id=(
                                tokenizer.pad_token_id
                                if tokenizer.pad_token_id is not None
                                else tokenizer.eos_token_id
                            ),
                        )
                else:
                    output_ids = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=self.max_new_tokens,
                        pad_token_id=(
                            tokenizer.pad_token_id
                            if tokenizer.pad_token_id is not None
                            else tokenizer.eos_token_id
                        ),
                    )
        finally:
            self.generate_elapsed_ms += (perf_counter() - generate_started) * 1000

        generated_ids = output_ids[0, input_ids.shape[1] :]
        raw_completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completion = raw_completion.strip()
        parsed_tool_calls = _parse_tool_call_details(
            raw_completion,
            call_id=f"hf-call-{len(messages)}",
        )
        tool_calls = parsed_tool_calls.calls
        self.parsed_tool_call_count += len(tool_calls)
        self.tool_parse_error_count += parsed_tool_calls.errors

        if len(tool_calls) > 1 and self.capture_activation:
            # A single generation-prefill state cannot be attributed to each action in a
            # multi-call assistant turn. Keep parsing/execution support for no-defense runs,
            # but make activation-based defenses invalid rather than silently reusing it.
            capture = None
            state["agent_defense.activation_error"] = "multiple_tool_calls_require_per_call_activations"

        selected_ids = input_ids
        selected_mask = attention_mask
        if len(tool_calls) == 1 and self.capture_activation and self.position == "function_call":
            try:
                tool_call_length = _tool_call_end_length(
                    tokenizer,
                    generated_ids,
                    raw_completion,
                    target_char=parsed_tool_calls.spans[0][1],
                )
                call_ids = generated_ids[:tool_call_length].unsqueeze(0)
                full_ids = torch.cat([input_ids, call_ids], dim=1)
                full_mask = torch.cat([attention_mask, torch.ones_like(call_ids)], dim=1)
                with (
                    torch.inference_mode(),
                    _ResidualPreCapture(
                        model,
                        self.layer,
                        full_mask,
                    ) as capture,
                ):
                    model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
                self.replay_forward_count += 1
                selected_ids = full_ids
                selected_mask = full_mask
            except ValueError:
                capture = None
                state["agent_defense.activation_error"] = "function_call_replay_boundary_error"

        if tool_calls and capture is not None and capture.value is not None:
            last_index = int(selected_mask[0].nonzero()[-1])
            token_id = int(selected_ids[0, last_index])
            state["agent_defense.activation"] = capture.value
            state["agent_defense.activation_metadata"] = {
                **self.activation_metadata,
                "model_id": self.display_model_id,
                "revision": _resolved_revision(model, self.revision),
                "model_dtype": str(next(model.parameters()).dtype),
                "quantization_config_hash": _quantization_config_hash(model),
                "tokenizer_class": type(tokenizer).__name__,
                "layer": capture.layer,
                "module_path": f"{capture.module_path}.{capture.layer}",
                "state_kind": "resid_pre",
                "position": _CAPTURE_POSITION_NAMES[self.position],
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]),
                "chat_template_hash": _chat_template_hash(tokenizer),
                "tool_schema_hash": _tool_schema_hash(tools),
                "system_message_hash": _system_message_hash(messages),
                "render_mode": render_mode,
                "generation_seed": self.seed,
                "extra_forward_count": int(self.position == "function_call"),
            }
            if checkpoint_content_id := self.checkpoint_content_id():
                state["agent_defense.activation_metadata"]["checkpoint_content_id"] = checkpoint_content_id
        elif tool_calls and self.capture_activation:
            state.setdefault("agent_defense.activation_error", "capture_missing")

        assistant = ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(completion)],
            tool_calls=tool_calls,
        )
        return query, runtime, env, [*messages, assistant], state
