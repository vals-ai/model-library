# Document truncation retries

`model_library.query_utils.query_with_truncation_retry` shortens document text
to recover from context and output token limits. Fixed instructions supplied by
`build_prompt` are retained. Successful calls return the query result and counts
of each kind of truncation performed.

The empty document is still attempted. If that query raises
`MaxContextWindowExceededError` or `MaxOutputTokensExceededError`, the exception
propagates to the caller. A `MAX_TOKENS` finish reason on the empty document raises
`MaxOutputTokensExceededError` with the raw finish reason. Further document
truncation cannot help, so the helper does not repeat that request.
