import torch
import torch.nn.functional as F
import math
import sys 
sys.path.append("..")  # Adjust the path as needed 
from src.utils.utils import (normalize_residual,
                        denormalize_residual, normalize_to_minus1_1_, denormalize_from_minus1_1_)

# def normalize_to_minus1_1_(x, ref_min=None, ref_max=None):
#     B = x.size(0)
#     dims = list(range(1, x.dim()))  # normalize over all but batch
#     if ref_min is None or ref_max is None:
#         x_min = x.view(B, -1).min(dim=1)[0].view(B, *[1]*(x.dim()-1))
#         x_max = x.view(B, -1).max(dim=1)[0].view(B, *[1]*(x.dim()-1))
#     else:
#         x_min = ref_min
#         x_max = ref_max
#     scale = x_max - x_min
#     scale[scale == 0] = 1.0
#     x.sub_(x_min).div_(scale).mul_(2).sub_(1)
#     return x_min, x_max

# --- Test Cases ---

def test_normalization_function():
    print("Running tests for normalize_to_minus1_1_...\n")

    # Test Case 1: Single batch item, simple range
    print("Test Case 1: Single batch item, simple range (0-10)")
    data1 = torch.tensor([[[[
        [0.0, 2.0, 4.0],
        [6.0, 8.0, 10.0]
    ]]]]) # Shape: (1, 1, 1, 2, 3)
    
    data2 = data1.clone() # Keep a copy of original 
    
    
    original_data1 = data1.clone() # Keep a copy of original
    
    norm_min1, norm_max1 = normalize_to_minus1_1_(data1)
    
    # Expected output: 0->-1, 10->1, 5->0, etc.
    # (val - min) / (max - min) * 2 - 1
    # (0 - 0) / 10 * 2 - 1 = -1
    # (10 - 0) / 10 * 2 - 1 = 1
    # (5 - 0) / 10 * 2 - 1 = 0
    expected_data1 = torch.tensor([[[[
        [-1.0, -0.6, -0.2],
        [0.2, 0.6, 1.0]
    ]]]])
    
    # Normalize the data
    torch.testing.assert_close(data2, expected_data1)
    
    normalize_to_minus1_1_(data2, norm_min1, norm_max1)
    

    torch.testing.assert_close(data2, expected_data1)
    # print(f"  Normalized data1 (expected): {expected_data1}")
    # print(f"  Normalized data1 (actual):   {data2}")
    # print(f"  Min/Max for data1: {norm_min1.item():.1f}, {norm_max1.item():.1f}")
    # assert norm_min1.item() == original_data1.min().item()
    # assert norm_max1.item() == original_data1.max().item()
    # print("  Test Case 1 Passed.\n")

    # Test Case 2: Multiple batch items, different ranges
#     print("Test Case 2: Multiple batch items, different ranges")
#     data2 = torch.tensor(
#         [[[[[0.0, 5.0, 10.0]]]],  # Batch 0: range 0-10
#         [[[[[-5.0, 0.0, 5.0]]]]], # Batch 1: range -5-5
#         [[[[[100.0, 150.0, 200.0]]]]] # Batch 2: range 100-200
# ) # Shape: (3, 1, 1, 1, 3)

#     original_data2 = data2.clone()
    
#     norm_min2, norm_max2 = normalize_to_minus1_1_(data2)

#     expected_data2 = torch.tensor([
#         [[[[[-1.0, 0.0, 1.0]]]],
#         [[[[[-1.0, 0.0, 1.0]]]],
#         [[[[[-1.0, 0.0, 1.0]]]]]
#     ])

#     torch.testing.assert_close(data2, expected_data2)
#     print(f"  Normalized data2 (expected):\n{expected_data2}")
#     print(f"  Normalized data2 (actual):\n{data2}")
#     print(f"  Min/Max for data2:\n{norm_min2.squeeze()}, {norm_max2.squeeze()}")
    
#     # Verify min/max for each batch item
#     assert norm_min2[0].item() == original_data2[0].min().item()
#     assert norm_max2[0].item() == original_data2[0].max().item()
#     assert norm_min2[1].item() == original_data2[1].min().item()
#     assert norm_max2[1].item() == original_data2[1].max().item()
#     assert norm_min2[2].item() == original_data2[2].min().item()
#     assert norm_max2[2].item() == original_data2[2].max().item()
#     print("  Test Case 2 Passed.\n")

    # # Test Case 3: Using reference min/max (second call in your training loop)
    # print("Test Case 3: Using reference min/max")
    # # Simulate 'inputs' data
    # inputs_data = torch.tensor([[[[
    #     [0.0, 2.0, 4.0],
    #     [6.0, 8.0, 10.0]
    # ]]]])
    
    # # First call, get min/max from inputs
    # _, input_min_ref, input_max_ref = normalize_to_minus1_1_(inputs_data.clone()) # Clone inputs to keep original if needed
    # print(f"  Reference Min/Max from inputs: {input_min_ref.item()}, {input_max_ref.item()}")

    # # Simulate 'targets' data, which should be normalized using inputs' min/max
    # targets_data = torch.tensor([[[[
    #     [1.0, 3.0, 5.0],
    #     [7.0, 9.0, 11.0] # Notice 11.0 is outside inputs' range 0-10
    # ]]]])
    
    # original_targets_data = targets_data.clone()

    # # Normalize targets using the reference min/max from inputs
    # # The function returns the min/max used for normalization *within that call*,
    # # which in this case will be the ref_min/max themselves.
    # norm_min3, norm_max3 = normalize_to_minus1_1_(targets_data, input_min_ref, input_max_ref)
    
    # # Expected: Normalized based on 0-10 scale
    # # (1 - 0) / 10 * 2 - 1 = -0.8
    # # (5 - 0) / 10 * 2 - 1 = 0.0
    # # (11 - 0) / 10 * 2 - 1 = 1.2 (can be outside -1, 1 if ref range is smaller)
    # expected_targets_data = torch.tensor([[[[
    #     [-0.8, -0.4, 0.0],
    #     [0.4, 0.8, 1.2]
    # ]]]])

    # torch.testing.assert_close(targets_data, expected_targets_data)
    # print(f"  Normalized targets (expected):\n{expected_targets_data}")
    # print(f"  Normalized targets (actual):\n{targets_data}")
    # # Verify that the returned min/max are the ref_min/max
    # assert norm_min3.item() == input_min_ref.item()
    # assert norm_max3.item() == input_max_ref.item()
    # print("  Test Case 3 Passed.\n")

    # # Test Case 4: Data with min == max (e.g., all zeros or all ones)
    # print("Test Case 4: Data with min == max")
    # data4_zeros = torch.zeros(1, 1, 2, 2, 2)
    # original_data4_zeros = data4_zeros.clone()
    # nm4_min, nm4_max = normalize_to_minus1_1_(data4_zeros)
    # # Expected: all -1.0 when scale is 0 and becomes 1.0 (0-0)/1*2-1 = -1
    # torch.testing.assert_close(data4_zeros, -torch.ones_like(data4_zeros))
    # print(f"  Normalized all zeros (expected):\n{-torch.ones_like(original_data4_zeros)}")
    # print(f"  Normalized all zeros (actual):\n{data4_zeros}")
    # print(f"  Min/Max for all zeros: {nm4_min.item()}, {nm4_max.item()}")
    # assert nm4_min.item() == 0.0
    # assert nm4_max.item() == 0.0
    # print("  Test Case 4a (all zeros) Passed.\n")

    # data4_ones = torch.ones(1, 1, 2, 2, 2) * 5.0 # All 5s
    # original_data4_ones = data4_ones.clone()
    # nm4_min_ones, nm4_max_ones = normalize_to_minus1_1_(data4_ones)
    # torch.testing.assert_close(data4_ones, -torch.ones_like(data4_ones))
    # print(f"  Normalized all fives (expected):\n{-torch.ones_like(original_data4_ones)}")
    # print(f"  Normalized all fives (actual):\n{data4_ones}")
    # print(f"  Min/Max for all fives: {nm4_min_ones.item()}, {nm4_max_ones.item()}")
    # assert nm4_min_ones.item() == 5.0
    # assert nm4_max_ones.item() == 5.0
    # print("  Test Case 4b (all constant) Passed.\n")

    # # Test Case 5: Larger 3D tensor
    # print("Test Case 5: Larger 3D tensor (2x1x4x4x4)")
    # data5 = torch.rand(2, 1, 4, 4, 4) * 100 - 50 # Random values between -50 and 50
    # original_data5_clone = data5.clone()

    # norm_min5, norm_max5 = normalize_to_minus1_1_(data5)

    # # Verify that the first item is normalized correctly
    # expected_norm_item0 = (original_data5_clone[0] - original_data5_clone[0].min()) / \
    #                       (original_data5_clone[0].max() - original_data5_clone[0].min()) * 2 - 1
    
    # torch.testing.assert_close(data5[0], expected_norm_item0)
    # print(f"  Normalized data5 (first batch item) matches manual calculation.")
    # print(f"  Min/Max for data5 batch 0: {norm_min5[0].item():.2f}, {norm_max5[0].item():.2f}")
    # print(f"  Min/Max for data5 batch 1: {norm_min5[1].item():.2f}, {norm_max5[1].item():.2f}")
    
    # # Check that overall min/max are -1/1 for normalized parts
    # assert data5.min().item() >= -1.0 - 1e-6 # Allowing for small floating point errors
    # assert data5.max().item() <= 1.0 + 1e-6 # Allowing for small floating point errors

    # print("  Test Case 5 Passed.\n")


    # print("All tests passed!")

# Run the tests
if __name__ == "__main__":
    test_normalization_function()