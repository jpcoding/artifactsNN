import numpy as np

def extract_random_paired_blocks(original, quantized, block_size=64, num_blocks=5, seed=1234):
    assert original.shape == quantized.shape, "Original and quantized data must have the same shape"
    assert original.ndim == 3, "Expected 3D volume input"
    D, H, W = original.shape

    np.random.seed(seed)  # Ensure repeatability
    blocks_orig, blocks_quant = [], []

    for _ in range(num_blocks):
        d = np.random.randint(0, D - block_size + 1)
        h = np.random.randint(0, H - block_size + 1)
        w = np.random.randint(0, W - block_size + 1)

        block_orig = original[d:d+block_size, h:h+block_size, w:w+block_size]
        block_quant = quantized[d:d+block_size, h:h+block_size, w:w+block_size]

        blocks_orig.append(block_orig)
        blocks_quant.append(block_quant)

    return blocks_orig, blocks_quant
