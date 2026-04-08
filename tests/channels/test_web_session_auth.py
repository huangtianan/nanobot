"""Web session auth helpers."""

from nanobot.channels.web_session_auth import bearer_token_from_message_metadata, web_session_auth_settings


def test_bearer_from_metadata_root() -> None:
    assert bearer_token_from_message_metadata({"auth_token": " abc "}) == "abc"


def test_bearer_from_nested_data_agent() -> None:
    assert bearer_token_from_message_metadata({"data_agent": {"authToken": "t1"}}) == "t1"


def test_web_session_auth_settings_dict() -> None:
    base, path, to = web_session_auth_settings(
        {"sessionAuthBaseUrl": "https://auth.example", "sessionAuthMePath": "/me", "sessionAuthTimeout": 5}
    )
    assert base == "https://auth.example"
    assert path == "/me"
    assert to == 5.0


def test_web_session_auth_settings_none() -> None:
    base, path, to = web_session_auth_settings(None)
    assert base == ""
    assert path == "/auth/users/me"
    assert to == 10.0
