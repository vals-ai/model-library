from model_gateway.run_token_types import ModelPermissions, RUN_TOKEN_OPERATION_PATHS


def intersect_permissions(
    token: ModelPermissions, current: ModelPermissions
) -> ModelPermissions:
    token_operations = token.allowed_operations
    current_operations = current.allowed_operations
    if token_operations is None:
        if current_operations is None:
            operations = None
        else:
            operations = list(
                dict.fromkeys(
                    operation
                    for operation in current_operations
                    if operation in RUN_TOKEN_OPERATION_PATHS
                )
            )
    else:
        allowed = (
            RUN_TOKEN_OPERATION_PATHS.keys()
            if current_operations is None
            else set(current_operations)
        )
        operations = list(
            dict.fromkeys(
                operation for operation in token_operations if operation in allowed
            )
        )
    models = token.allowed_models
    if models is not None and current.allowed_models is not None:
        current_models = set(current.allowed_models)
        models = list(dict.fromkeys(m for m in models if m in current_models))
    return ModelPermissions(
        allowed_models=models,
        # Managed roots have no embedding scope; the child token sets it.
        allowed_embedding_models=token.allowed_embedding_models,
        allowed_operations=operations,
    )
