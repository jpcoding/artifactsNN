

def normalize_to_minus1_1_(x, ref_min=None, ref_max=None):
    B = x.size(0)
    dims = list(range(1, x.dim()))  # normalize over all but batch
    if ref_min is None or ref_max is None:
        x_min = x.view(B, -1).min(dim=1)[0].view(B, *[1]*(x.dim()-1))
        x_max = x.view(B, -1).max(dim=1)[0].view(B, *[1]*(x.dim()-1))
    else:
        x_min = ref_min
        x_max = ref_max
    scale = x_max - x_min
    scale[scale == 0] = 1.0
    x.sub_(x_min).div_(scale).mul_(2).sub_(1)
    return x_min, x_max


def denormalize_from_minus1_1_(x, x_min, x_max):
    return x.add_(1).mul_(0.5).mul_(x_max - x_min).add_(x_min)


def normalize_error(x):
    B = x.size(0)
    x_reshaped = x.view(B, -1)
    abs_max = x_reshaped.abs().max(dim=1)[0].view(B, *[1]*(x.dim()-1))
    abs_max[abs_max == 0] = 1.0
    return x / abs_max


def normalize_residual(x, bound):
    bound = bound.view(-1, *[1]*(x.dim()-1))  # Expand to match x shape
    bound = bound.clamp(min=1e-8)
    return x / bound


def denormalize_residual(x, bound):
    bound = bound.view(-1, *[1]*(x.dim()-1))
    return x * bound
