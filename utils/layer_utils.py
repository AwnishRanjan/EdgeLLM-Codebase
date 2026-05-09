import math


def get_base_model(model):
    """Unwrap PEFT/model containers until the decoder with layers is reached."""
    base = model
    while hasattr(base, "model") and not hasattr(base, "layers"):
        base = base.model
    return base


def get_num_layers(model):
    base = get_base_model(model)
    if not hasattr(base, "layers"):
        raise ValueError("Could not find decoder layers on the model.")
    return len(base.layers)


def get_exit_layer_indices(num_layers, num_exits=4, zero_based=True):
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")

    exits = []
    for index in range(1, min(num_exits, num_layers) + 1):
        layer = math.ceil(num_layers * index / min(num_exits, num_layers))
        if layer not in exits:
            exits.append(layer)

    return [layer - 1 for layer in exits] if zero_based else exits


def get_stride_exit_layer_indices(num_layers, stride_size, zero_based=True):
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if stride_size <= 0:
        raise ValueError("stride_size must be positive.")

    exits = list(range(stride_size, num_layers + 1, stride_size))
    if not exits or exits[-1] != num_layers:
        exits.append(num_layers)

    return [layer - 1 for layer in exits] if zero_based else exits
