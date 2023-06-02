import numpy as np

def fill_array_block_cyclic(global_shape, local_shape):
    global_rows, global_cols = global_shape
    local_rows, local_cols = local_shape

    # Calculate the number of blocks
    num_blocks = (global_rows // local_rows) * (global_cols // local_cols)

    # Create the global array
    global_array = np.empty(global_shape)

    # Fill the array in block cyclic manner
    block_idx = 0
    counter = 0
    for i in range(0, global_rows, local_rows):
        for j in range(0, global_cols, local_cols):
            # Calculate the local block indices
            local_row_idx, local_col_idx = np.unravel_index(block_idx, (global_rows // local_rows, global_cols // local_cols))

            # Fill the local block
            local_array = np.full((local_rows, local_cols), counter)
            global_array[i:i+local_rows, j:j+local_cols] = local_array

            block_idx += 1
            counter += 1

    return global_array

# Set the global shape and local shape
global_shape = (10, 10)
local_shape = (2, 2)

# Fill the array using block cyclic distribution
global_array = fill_array_block_cyclic(global_shape, local_shape)

# Print the global array
print(global_array)