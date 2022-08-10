import numpy as np

def detA_2x2(a, b, c, d):
    return a*d-b*c

def inv(A):

    if len(A.shape) == 1:
        return ValueError('Matix must have at least 2 dimensions!')
    if A.shape[1] not in [2, 3]:
        return ValueError('Matrix must be either 2x2 or 3x3!')

    elif len(A.shape) == 2:
        if A.shape[0] == 2:       # 2x2
            detA = detA_2x2(A[0, 0], A[0,1], A[1,0], A[1,1])
            cofactor = np.array([[A[1,1], -A[0,1]],
                                [-A[1,0], A[0,0]]])
            Ainv = cofactor/detA
        else:                     # 3x3
            detA = A[0,0]*detA_2x2(A[1,1],A[1,2],A[2,1],A[2,2]) + \
                    -A[0,1]*detA_2x2(A[1,0],A[1,2],A[2,0],A[2,2]) + \
                    +A[0,2]*detA_2x2(A[1,0],A[1,1],A[2,0],A[2,1])

            cf00 = detA_2x2(A[1,1],A[1,2],A[2,1],A[2,2])
            cf01 = detA_2x2(A[0,2],A[0,1],A[2,2],A[2,1])
            cf02 = detA_2x2(A[0,1],A[0,2],A[1,1],A[1,2])
            cf10 = detA_2x2(A[1,2],A[1,0],A[2,2],A[2,0])
            cf11 = detA_2x2(A[0,0],A[0,2],A[2,0],A[2,2])
            cf12 = detA_2x2(A[0,2],A[0,0],A[1,2],A[1,0])
            cf20 = detA_2x2(A[1,0],A[1,1],A[2,0],A[2,1])
            cf21 = detA_2x2(A[0,1],A[0,0],A[2,1],A[2,0])
            cf22 = detA_2x2(A[0,0],A[0,1],A[1,0],A[1,1])

            Ainv = np.array([[cf00, cf01, cf02],
                            [cf10, cf11, cf12],
                            [cf20, cf21, cf22]])/detA

    elif len(A.shape) == 3:
        Ainv = np.zeros_like(A)

        if A.shape[1] == 2:       # 2x2
            detA = detA_2x2(A[:,0, 0], A[:,0,1], A[:,1,0], A[:,1,1])

            cf00 = A[:,1,1]
            cf01 = -A[:,0,1]
            cf10 = -A[:,1,0]
            cf11 = A[:,0,0]

            Ainv[:,0,0] = cf00
            Ainv[:,0,1] = cf01
            Ainv[:,1,0] = cf10
            Ainv[:,1,1] = cf11

            Ainv = Ainv/detA[:,None,None]
        else:                     # 3x3
            detA = A[:,0,0]*detA_2x2(A[:,1,1],A[:,1,2],A[:,2,1],A[:,2,2]) + \
                    -A[:,0,1]*detA_2x2(A[:,1,0],A[:,1,2],A[:,2,0],A[:,2,2]) + \
                    +A[:,0,2]*detA_2x2(A[:,1,0],A[:,1,1],A[:,2,0],A[:,2,1])

            cf00 = detA_2x2(A[:,1,1],A[:,1,2],A[:,2,1],A[:,2,2])
            cf01 = detA_2x2(A[:,0,2],A[:,0,1],A[:,2,2],A[:,2,1])
            cf02 = detA_2x2(A[:,0,1],A[:,0,2],A[:,1,1],A[:,1,2])
            cf10 = detA_2x2(A[:,1,2],A[:,1,0],A[:,2,2],A[:,2,0])
            cf11 = detA_2x2(A[:,0,0],A[:,0,2],A[:,2,0],A[:,2,2])
            cf12 = detA_2x2(A[:,0,2],A[:,0,0],A[:,1,2],A[:,1,0])
            cf20 = detA_2x2(A[:,1,0],A[:,1,1],A[:,2,0],A[:,2,1])
            cf21 = detA_2x2(A[:,0,1],A[:,0,0],A[:,2,1],A[:,2,0])
            cf22 = detA_2x2(A[:,0,0],A[:,0,1],A[:,1,0],A[:,1,1])

            Ainv[:,0,0] = cf00
            Ainv[:,0,1] = cf01
            Ainv[:,0,2] = cf02
            Ainv[:,1,0] = cf10
            Ainv[:,1,1] = cf11
            Ainv[:,1,2] = cf12
            Ainv[:,2,0] = cf20
            Ainv[:,2,1] = cf21
            Ainv[:,2,2] = cf22

            Ainv = Ainv/detA[:,None,None]

    return Ainv, detA

if __name__ == '__main__':
    A_2x2 = np.random.rand(2,2)
    A_3x3 = np.random.rand(3,3)


    print(np.linalg.norm((np.linalg.inv(A_2x2) - inv(A_2x2))))
    print(np.linalg.norm((np.linalg.inv(A_3x3) - inv(A_3x3))))

    a = np.array([[[4, 8, 2],
                    [5, 0, 2],
                    [1, 1, 3]],
                    
                    [[7, 0, 8],
                    [9, 5, 4],
                    [3, 4, 2]]])

    b = np.array([[[4, 8],
                    [5, 0]],
                    
                    [[7, 0],
                    [9, 5]]])

    #2x2
    i = np.zeros_like(a, dtype=float)
    i[0,:,:] = np.linalg.inv(a[0,:,:])
    i[1,:,:] = np.linalg.inv(a[1,:,:])

    i_custom = inv(a)

    print(np.linalg.norm(i-i_custom))

    #3x3
    i = np.zeros_like(b, dtype=float)
    i[0,:,:] = np.linalg.inv(b[0,:,:])
    i[1,:,:] = np.linalg.inv(b[1,:,:])

    i_custom = inv(b)

    print(np.linalg.norm(i-i_custom))
