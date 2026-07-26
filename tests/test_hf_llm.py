from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from agentdojo.functions_runtime import FunctionCall, make_function
from agentdojo.types import ChatAssistantMessage, text_content_block_from_string

from agent_defense.artifacts import DetectorArtifact
from agent_defense.hf_llm import (
    HuggingFaceToolCallingLLM,
    _chat_template_hash,
    _display_model_id,
    _local_checkpoint_content_id,
    _parse_tool_call_details,
    _parse_tool_call_with_diagnostics,
    _ResidualPreCapture,
    _structured_tool_markup_count,
    _tool_call_end_length,
    parse_tool_call,
    render_agentdojo_messages,
    render_native_agentdojo_messages,
)


def test_parser_supports_multiple_agentdojo_calls() -> None:
    calls = parse_tool_call(
        '<function=read_file>{"file_path":"a.txt"}</function>\n'
        '<function=send_email>{"to":"a@example.com","body":"ok"}</function>',
        call_id="candidate",
    )

    assert [call.function for call in calls] == ["read_file", "send_email"]
    assert [call.id for call in calls] == ["candidate", "candidate-1"]


def test_parser_supports_qwen_arguments_as_object_or_json_string() -> None:
    calls = parse_tool_call(
        '<tool_call>{"name":"read_file","arguments":{"file_path":"a.txt"}}</tool_call>\n'
        '<tool_call>{"name":"send_email","arguments":"{\\"to\\":\\"a@example.com\\"}"}</tool_call>'
    )

    assert [call.function for call in calls] == ["read_file", "send_email"]
    assert calls[0].args["file_path"] == "a.txt"
    assert calls[1].args["to"] == "a@example.com"


def test_parser_rejects_malformed_or_non_object_arguments() -> None:
    assert parse_tool_call("<tool_call>not-json</tool_call>") == []
    assert parse_tool_call('<function=read_file>["a.txt"]</function>') == []
    assert _structured_tool_markup_count("<tool_call>not-json</tool_call>") == 1
    assert _structured_tool_markup_count("<function=read_file>unterminated") == 1


def test_parser_diagnostics_do_not_flag_mixed_valid_syntaxes_or_literal_markers() -> None:
    mixed = (
        '<function=read_file>{"file_path":"a.txt"}</function>'
        '<tool_call>{"name":"send_email","arguments":{"to":"a@example.com"}}</tool_call>'
    )
    calls, errors = _parse_tool_call_with_diagnostics(mixed)

    assert [call.function for call in calls] == ["read_file", "send_email"]
    assert errors == 0

    literal_marker = (
        '<tool_call>{"name":"write_file","arguments":'
        '{"content":"the literal marker is <tool_call>"}}</tool_call>'
    )
    calls, errors = _parse_tool_call_with_diagnostics(literal_marker)

    assert [call.function for call in calls] == ["write_file"]
    assert errors == 0


def test_rendering_does_not_duplicate_raw_and_structured_tool_markup() -> None:
    message = ChatAssistantMessage(
        role="assistant",
        content=[
            text_content_block_from_string(
                'I will call it.\n<tool_call>{"name":"add","arguments":{"x":2,"y":3}}</tool_call>'
            )
        ],
        tool_calls=[FunctionCall(function="add", args={"x": 2, "y": 3}, id="call-1")],
    )

    fallback = render_agentdojo_messages([message], ())
    native = render_native_agentdojo_messages([message])

    assert fallback[0]["content"].count("<function=add>") == 1
    assert "<tool_call>" not in fallback[0]["content"]
    assert native[0]["content"] == "I will call it."
    assert len(native[0]["tool_calls"]) == 1


def test_disable_thinking_is_forwarded_to_native_tool_template_and_render_identity() -> None:
    calls: list[dict[str, object]] = []

    class StubTokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs) -> str:
            calls.append({"messages": messages, **kwargs})
            return "native rendered prompt"

    def add(x: int, y: int) -> int:
        """Add two integers.

        :param x: First integer.
        :param y: Second integer.
        """

        return x + y

    native_messages = [{"role": "user", "content": "Add 2 and 3."}]
    fallback_messages = [{"role": "user", "content": "fallback"}]
    llm = HuggingFaceToolCallingLLM("org/model", disable_thinking=True)

    rendered, render_mode = llm._render(
        StubTokenizer(),
        native_messages,
        fallback_messages,
        (make_function(add),),
    )

    assert rendered == "native rendered prompt"
    assert render_mode == "native_tools:disable_thinking"
    assert len(calls) == 1
    assert calls[0]["messages"] is native_messages
    assert calls[0]["enable_thinking"] is False
    assert calls[0]["tokenize"] is False
    assert calls[0]["add_generation_prompt"] is True
    assert calls[0]["tools"]


def test_function_call_replay_stops_at_original_closing_tag_before_prose_or_eos() -> None:
    class CharacterTokenizer:
        @staticmethod
        def decode(ids, *, skip_special_tokens: bool) -> str:
            del skip_special_tokens
            return "".join(chr(int(value)) for value in ids)

    raw = '<tool_call>{"name":"add","arguments":{}}</tool_call> trailing prose\x00'
    generated_ids = np.asarray([ord(character) for character in raw])

    length = _tool_call_end_length(CharacterTokenizer(), generated_ids, raw)

    expected = raw.index("</tool_call>") + len("</tool_call>")
    assert length == expected
    assert "".join(chr(value) for value in generated_ids[:length]).endswith("</tool_call>")

    with pytest.raises(ValueError, match="closing tool-call tag"):
        _tool_call_end_length(CharacterTokenizer(), generated_ids, "plain text only")


def test_function_call_replay_uses_selected_valid_span_not_a_later_extra_closing_tag() -> None:
    class CharacterTokenizer:
        @staticmethod
        def decode(ids, *, skip_special_tokens: bool) -> str:
            del skip_special_tokens
            return "".join(chr(int(value)) for value in ids)

    raw = '<tool_call>{"name":"add","arguments":{}}</tool_call> prose </tool_call>'
    generated_ids = np.asarray([ord(character) for character in raw])
    parsed = _parse_tool_call_details(raw)

    assert len(parsed.calls) == 1
    length = _tool_call_end_length(
        CharacterTokenizer(),
        generated_ids,
        raw,
        target_char=parsed.spans[0][1],
    )

    assert length == raw.index("</tool_call>") + len("</tool_call>")


def test_residual_pre_capture_selects_last_non_padding_token_and_removes_hook() -> None:
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def forward(self, hidden):
            return hidden + 1

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Block()])

        def forward(self, hidden):
            return self.model.layers[0](hidden)

    model = ToyModel()
    mask = torch.tensor([[1, 1, 0]])
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
    block = model.model.layers[0]
    hooks_before = len(block._forward_pre_hooks)

    with _ResidualPreCapture(model, 0, mask) as capture:
        model(hidden)

    assert np.allclose(capture.value, [3.0, 4.0])
    assert len(block._forward_pre_hooks) == hooks_before
    assert capture.module_path == "model.layers"
    assert SimpleNamespace(value=capture.layer).value == 0


def test_artifact_preflight_checks_real_model_identity_without_downloading() -> None:
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def forward(self, hidden):
            return hidden

    class StubTokenizer:
        chat_template = "{{ messages }}"

    class StubModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Block(), Block()])
            self.config = SimpleNamespace(
                hidden_size=3,
                quantization_config=None,
                _commit_hash="revision-a",
            )

    llm = HuggingFaceToolCallingLLM("org/model", layer=-1, position="tool_input", revision="mutable-main")
    llm._model = StubModel()
    llm._tokenizer = StubTokenizer()
    artifact = DetectorArtifact(
        kind="linear_probe",
        weights=(0.1, 0.2, 0.3),
        bias=0.0,
        threshold=0.5,
        model_id="org/model",
        layer=1,
        position="generation_prefill_last_nonpad",
        metadata={
            "activation_compatibility": {
                "revision": "revision-a",
                "model_dtype": "torch.float32",
                "quantization_config_hash": "none",
                "tokenizer_class": "StubTokenizer",
                "state_kind": "resid_pre",
                "module_path": "model.layers.1",
                "chat_template_hash": _chat_template_hash(llm._tokenizer),
            }
        },
    )

    result = llm.preflight_artifact(artifact)

    assert result["dimension"] == 3
    assert result["layer"] == 1
    assert result["position"] == "generation_prefill_last_nonpad"
    assert result["revision"] == "revision-a"
    assert "checkpoint_content_id" not in result

    with pytest.raises(ValueError, match="position"):
        llm.preflight_artifact(replace(artifact, position="function_call_end"))
    with pytest.raises(ValueError, match="dimension"):
        llm.preflight_artifact(replace(artifact, weights=(0.1, 0.2)))

    llm.layer = 99
    with pytest.raises(ValueError, match="outside a model"):
        llm.preflight_artifact(artifact)

    llm.layer = -1
    with pytest.raises(ValueError, match="missing preflight compatibility"):
        llm.preflight_artifact(replace(artifact, metadata={}))


def _write_local_checkpoint(path, *, config_marker: str, weight_marker: int) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        f'{{"architectures":["StubModel"],"marker":"{config_marker}"}}',
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        '{"chat_template":"{{ messages }}"}',
        encoding="utf-8",
    )
    (path / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":262144},"weight_map":{"w":"model.safetensors"}}',
        encoding="utf-8",
    )
    (path / "model.safetensors").write_bytes(bytes([weight_marker]) * (256 * 1024))


def test_local_checkpoint_content_id_is_path_independent_and_content_sensitive(tmp_path) -> None:
    first = tmp_path / "first-private-root" / "same-model"
    second = tmp_path / "second-private-root" / "same-model"
    _write_local_checkpoint(first, config_marker="a", weight_marker=1)
    _write_local_checkpoint(second, config_marker="a", weight_marker=1)

    first_id = _local_checkpoint_content_id(str(first))
    second_id = _local_checkpoint_content_id(str(second))

    assert first_id == second_id
    assert first_id is not None and first_id.startswith("local-checkpoint-v1:")
    assert _local_checkpoint_content_id("org/remote-model") is None

    (second / "trainer_state.json").write_text('{"global_step":99}', encoding="utf-8")
    (second / "optimizer.pt").write_bytes(b"optimizer-state-that-is-not-loaded")
    (second / "training_args.bin").write_bytes(b"training-arguments")
    assert _local_checkpoint_content_id(str(second)) == first_id

    (second / "modeling_stub.py").write_text("MODEL_VERSION = 1\n", encoding="utf-8")
    assert _local_checkpoint_content_id(str(second)) == first_id
    trusted_code_id = _local_checkpoint_content_id(str(second), include_remote_code=True)
    (second / "modeling_stub.py").write_text("MODEL_VERSION = 2\n", encoding="utf-8")
    assert _local_checkpoint_content_id(str(second), include_remote_code=True) != trusted_code_id

    (second / "config.json").write_text('{"marker":"changed"}', encoding="utf-8")
    assert _local_checkpoint_content_id(str(second)) != first_id

    (second / "config.json").write_text(
        '{"architectures":["StubModel"],"marker":"a"}',
        encoding="utf-8",
    )
    with (second / "model.safetensors").open("r+b") as handle:
        handle.seek(128 * 1024)
        handle.write(b"changed-weight-sample")
    assert _local_checkpoint_content_id(str(second)) != first_id


def test_local_artifact_preflight_requires_matching_checkpoint_content_id(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def forward(self, hidden):
            return hidden

    class StubTokenizer:
        chat_template = "{{ messages }}"

    class StubModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Block(), Block()])
            self.config = SimpleNamespace(
                hidden_size=3,
                quantization_config=None,
                _commit_hash="revision-a",
            )

    checkpoint = tmp_path / "first" / "same-model"
    _write_local_checkpoint(checkpoint, config_marker="a", weight_marker=1)
    llm = HuggingFaceToolCallingLLM(str(checkpoint), layer=-1, position="tool_input")
    llm._model = StubModel()
    llm._tokenizer = StubTokenizer()
    content_id = _local_checkpoint_content_id(str(checkpoint))
    compatibility = {
        "checkpoint_content_id": content_id,
        "revision": "revision-a",
        "model_dtype": "torch.float32",
        "quantization_config_hash": "none",
        "tokenizer_class": "StubTokenizer",
        "state_kind": "resid_pre",
        "module_path": "model.layers.1",
        "chat_template_hash": _chat_template_hash(llm._tokenizer),
    }
    artifact = DetectorArtifact(
        kind="linear_probe",
        weights=(0.1, 0.2, 0.3),
        bias=0.0,
        threshold=0.5,
        model_id="same-model",
        layer=1,
        position="generation_prefill_last_nonpad",
        metadata={"activation_compatibility": compatibility},
    )

    result = llm.preflight_artifact(artifact)

    assert result["checkpoint_content_id"] == content_id
    legacy_compatibility = {
        key: value for key, value in compatibility.items() if key != "checkpoint_content_id"
    }
    with pytest.raises(ValueError, match="checkpoint_content_id"):
        llm.preflight_artifact(replace(artifact, metadata={"activation_compatibility": legacy_compatibility}))

    changed_checkpoint = tmp_path / "second" / "same-model"
    _write_local_checkpoint(changed_checkpoint, config_marker="different", weight_marker=1)
    changed_llm = HuggingFaceToolCallingLLM(str(changed_checkpoint), layer=-1, position="tool_input")
    changed_llm._model = StubModel()
    changed_llm._tokenizer = StubTokenizer()
    with pytest.raises(ValueError, match="checkpoint_content_id"):
        changed_llm.preflight_artifact(artifact)


def test_chat_template_hash_distinguishes_named_template_mappings() -> None:
    first = SimpleNamespace(chat_template={"default": "v1", "tool_use": "tools-v1"})
    second = SimpleNamespace(chat_template={"default": "v2", "tool_use": "tools-v1"})

    assert _chat_template_hash(first) != _chat_template_hash(second)


def test_loaded_backend_can_be_shared_without_triggering_load() -> None:
    model = object()
    tokenizer = object()
    llm = HuggingFaceToolCallingLLM("org/model")

    assert llm.loaded_backend() is None
    llm.attach_loaded_backend(model, tokenizer)

    assert llm.loaded_backend() == (model, tokenizer)
    with pytest.raises(ValueError, match="both model and tokenizer"):
        llm.attach_loaded_backend(model, None)


def test_display_model_id_does_not_expose_local_paths() -> None:
    assert _display_model_id("Qwen/Qwen3-8B") == "Qwen/Qwen3-8B"
    assert _display_model_id("/private/cache/Qwen3-8B") == "Qwen3-8B"
    assert _display_model_id(r"C:\private\cache\Qwen3-8B") == "Qwen3-8B"
