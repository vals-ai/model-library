import json

from model_library import model_library_settings


def default_aws_api_key() -> str:
    if getattr(model_library_settings, "AWS_ACCESS_KEY_ID", None):
        creds: dict[str, str] = {
            "AWS_ACCESS_KEY_ID": model_library_settings.AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": model_library_settings.AWS_SECRET_ACCESS_KEY,
            "AWS_DEFAULT_REGION": model_library_settings.AWS_DEFAULT_REGION,
        }
        session_token = model_library_settings.get("AWS_SESSION_TOKEN")
        if session_token:
            creds["AWS_SESSION_TOKEN"] = session_token
        return json.dumps(creds)
    return "using-environment"
