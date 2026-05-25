# Chat Web App

The chat web app is a local UI for trying registry models in the browser.

```bash
make install
make chat
```

Open `http://127.0.0.1:8000`, choose a model, and send messages. The server uses
the same environment variables and `model_library_settings` values as the Python
library, such as `OPENAI_API_KEY`.

The UI exposes:

- `/api/models` for model metadata from the registry
- `/api/chat` for one-shot chat requests
- `/ws/chat` for the browser conversation transport
