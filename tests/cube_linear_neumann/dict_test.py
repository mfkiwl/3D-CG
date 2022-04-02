import numpy as np

a = np.arange(30).reshape((10,3)).astype(np.int32)

di = {val.tobytes():key for key, val in enumerate(a)}
# print(di)
# print()
# for key in di:
#     print(type(key))
#     print(np.frombuffer(key, dtype=np.int32), di[key])

b = np.concatenate((a, np.array([50, 51, 52])[None,:]), axis=0).astype(np.int32)

for row in b:
    if row.tobytes() in di:
        print('here')
    else:
        print('not here')