import pytest

from agent_defense.demo import ATTACKER_IBAN, run_demo_trial, run_interview_sequence


@pytest.mark.integration
def test_demo_no_defense_allows_the_banking_attack() -> None:
    record, trace = run_demo_trial("none", attacked=True)

    assert record.utility_passed is True
    assert record.attack_succeeded is True
    assert record.malicious_tool_proposed is True
    assert record.malicious_tool_blocked is False
    assert any(
        item["tool"] == "send_money"
        and item["args"].get("recipient") == ATTACKER_IBAN
        and item["executed"] is True
        for item in trace
    )


@pytest.mark.integration
def test_demo_direction_defense_keeps_clean_utility_and_blocks_attack() -> None:
    clean, _ = run_demo_trial("direction", attacked=False)
    attacked, trace = run_demo_trial("direction", attacked=True)

    assert clean.utility_passed is True
    assert clean.normal_tool_blocked is False
    assert attacked.utility_passed is True
    assert attacked.attack_succeeded is False
    assert attacked.malicious_tool_blocked is True
    assert any(
        item["tool"] == "send_money"
        and item["args"].get("recipient") == ATTACKER_IBAN
        and item["decision"] == "block"
        and item["executed"] is False
        for item in trace
    )


@pytest.mark.integration
def test_interview_sequence_covers_before_after_and_clean_utility() -> None:
    cases = {item["case"]: item for item in run_interview_sequence()}

    assert cases["no_defense_attacked"]["record"].attack_succeeded is True
    assert cases["probe_attacked"]["record"].attack_succeeded is False
    assert cases["probe_clean"]["record"].utility_passed is True
    assert any(
        item["args"].get("recipient") == ATTACKER_IBAN
        and item["decision"] == "block"
        and item["executed"] is False
        for item in cases["probe_attacked"]["trace"]
    )
