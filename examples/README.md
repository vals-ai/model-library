# Model Library Examples

Example model use cases for the model_library.

## Structure

```
examples/
├── basics.py                    # Basic queries, system prompts, images, and files
├── images.py                    # Image handling (base64, file ID, URL)
├── files.py                     # File handling (base64, file ID, URL)
├── tool_calls.py                # Tool calling and multi-turn conversations
├── embeddings.py                # Embedding generation with OpenAI/Azure
├── setup.py                     # Setup utilities and logging configuration
├── data/
│   ├── images.py                # Sample image data (red square)
│   └── files.py                 # Sample file data (PDF)
└── advanced/
    ├── batch.py                 # Batch query operations
    ├── custom_retrier.py        # Custom retry logic and error handling
    ├── deep_research.py         # Deep research with web search and code interpreter
    └── stress.py                # Concurrent stress testing
```

## How to Run

use `python -m examples.[path_to.]file`

alternatively, run all examples at once with `make examples`.

optionally, add the model string you'd like to test after either command.

