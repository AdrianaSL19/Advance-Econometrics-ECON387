import numpy as np

np.random.seed(37)
tmp = np.random.rand(21, 21)  # generates a matrix tmp of 21×21

tmp[np.diag_indices_from(tmp)] = 1  # Change the diagonal elements of tmp to 1s
np.linalg.cond(tmp)  # Calculate condition number of tmp
np.linalg.inv(tmp)  # Calculate the inverse of tmp
np.trace(tmp)   # Calculate the trace of tmp
np.sort(tmp, 0) # Sort tmp across rows
tmp = tmp[:-1, :-1]  # Delete last row and column
tmp1 = np.reshape(tmp, (40, 10))
tmp2 = np.tile(tmp, (2, 2)) # generates a 40 by 40 array
np.linalg.cond(tmp2)  # Calculate condition number of tmp2
# print(np.linalg.inv(tmp2))  # Singular matrix
tmp2 = np.where(tmp2 <= 0, 0.5, tmp2)   # Change all non positive to 0.5
tmp2[0, 0] = -tmp2[0, 0]    # set first row/column element to its negative value
# tmp3 = np.log(tmp2) # calculates the natural log of tmp2
# np.argwhere(np.isnan(tmp3)) # Find the indices for NaN values which is [0, 0]
