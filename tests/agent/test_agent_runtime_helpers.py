from agent.agent_runtime_helpers import strip_think_blocks


class _Agent:
    pass


def test_strip_think_blocks_flattens_structured_assistant_content() -> None:
    content = [
        {"type": "output_text", "text": "<think>private</think>Visible"},
        {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
        {"type": "text", "text": " answer"},
    ]

    assert strip_think_blocks(_Agent(), content) == "Visible\n answer"
