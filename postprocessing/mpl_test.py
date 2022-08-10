# import numpy as np

# arr = np.array([2, 3, 4, 5, 3, 4, 5, 3, 5, 4, 7, 8, 3, 6, 2])
# count_arr = np.bincount(arr)

# print(count_arr)

# print(sum(count_arr[[3, 4]]))
# # Count occurrence of element '3' in numpy array
# print('Total occurences of "3" in array: ', count_arr[3])
# # Count occurrence of element '5' in numpy array
# print('Total occurences of "5" in array: ', count_arr[5])

# import matplotlib.pyplot as plt
# import numpy as np

# def discrete_matshow(data):
#     # get discrete colormap
#     cmap = plt.get_cmap('tab20', np.max(data) - np.min(data) + 1)
#     # cmap = plt.get_cmap('tab20')
#     # set limits .5 outside true range
#     mat = plt.matshow(data, cmap=cmap, vmin=np.min(data) - 0.5, 
#                       vmax=np.max(data) + 0.5)
#     # tell the colorbar to tick at integers
#     cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data) + 1))
#     # cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data)))
#     plt.show()

# # generate data
# a = np.random.randint(1, 9, size=(10, 10))
# discrete_matshow(a)

import matplotlib.pyplot as plt

cmap = plt.get_cmap('tab20', 20)    # PiYG
import numpy as np
for i in range(cmap.N):
    rgba = cmap(i)
    print(i+1, np.array(rgba[:-1]))