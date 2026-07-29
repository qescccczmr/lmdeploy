import pytest

from lmdeploy.serve.parsers import ResponseParserManager, validate_parser_names
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParserManager


@pytest.fixture
def default_response_parser_cls():
    cls = ResponseParserManager.get('default')
    reasoning_parser_cls = cls.reasoning_parser_cls
    tool_parser_cls = cls.tool_parser_cls
    try:
        cls.reasoning_parser_cls = None
        cls.tool_parser_cls = None
        yield cls
    finally:
        cls.reasoning_parser_cls = reasoning_parser_cls
        cls.tool_parser_cls = tool_parser_cls


def test_validate_parser_names_rejects_unknown_tool_parser_before_tokenizer():
    with pytest.raises(ValueError, match='The tool parser default is not in the parser list'):
        validate_parser_names(reasoning_parser_name='qwen-qwq', tool_parser_name='default')


def test_validate_parser_names_maps_legacy_reasoning_parser():
    reasoning_parser_name, tool_parser_name = validate_parser_names(
        reasoning_parser_name='qwen-qwq',
        tool_parser_name='interns2-preview',
        warn_legacy=False,
    )

    assert reasoning_parser_name == 'default'
    assert tool_parser_name == 'interns2-preview'


def test_response_parser_set_parsers_rejects_unknown_tool_parser(default_response_parser_cls):
    with pytest.raises(ValueError, match='The tool parser default is not in the parser list'):
        default_response_parser_cls.set_parsers(reasoning_parser_name='default', tool_parser_name='default')

    assert default_response_parser_cls.reasoning_parser_cls is None
    assert default_response_parser_cls.tool_parser_cls is None


def test_response_parser_set_parsers_accepts_registered_names(default_response_parser_cls):
    default_response_parser_cls.set_parsers(reasoning_parser_name='default', tool_parser_name='interns2-preview')

    assert default_response_parser_cls.reasoning_parser_cls is ReasoningParserManager.get('default')
    assert default_response_parser_cls.tool_parser_cls is ToolParserManager.get('interns2-preview')


def test_response_parser_accepts_thinking_alias_disabled(default_response_parser_cls):
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

    default_response_parser_cls.set_parsers(reasoning_parser_name='default')
    request = ChatCompletionRequest(
        model='moonshotai/Kimi-K2.6',
        messages=[],
        chat_template_kwargs={'thinking': False},
    )

    parser = default_response_parser_cls(request)
    content, tool_calls, reasoning_content = parser.parse_complete('北京')

    assert parser.enable_thinking is False
    assert content == '北京'
    assert tool_calls is None
    assert reasoning_content is None


def test_response_parser_accepts_thinking_alias_enabled(default_response_parser_cls):
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

    default_response_parser_cls.set_parsers(reasoning_parser_name='default')
    request = ChatCompletionRequest(
        model='moonshotai/Kimi-K2.6',
        messages=[],
        chat_template_kwargs={'thinking': True},
    )

    parser = default_response_parser_cls(request)
    content, tool_calls, reasoning_content = parser.parse_complete('分析过程</think>最终答案')

    assert parser.enable_thinking is True
    assert content == '最终答案'
    assert tool_calls is None
    assert reasoning_content == '分析过程'
