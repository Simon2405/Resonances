# author : Simon Ruget
# For the 1D linear chain with a four site defect, we compute the S-matrix



###########################
### Scattering function ###
###########################
function Scattering(z)
    ## Definingn constants 
    V = 2 # jump constant in the defect
    eps = 0.25 # coupling magnitude
    
    A = zeros((1,4)) # coupling between left part and defect
    A[1,1] = eps
    
    B = zeros((4,1)) # coupling between right part and defect
    B[4,1] = eps
    

    ## Defining the scattering matrix
    # We used Schur complement to express transmission matrix
    G0(z) = -1/z
    HV(z) = [[-z V 0 0]; [V -z V 0]; [0 V -z V]; [0 0 V -z]]

    # Finding propagative solution
    k = eigen([[z -1]; [1 0]]).values
    k2 = k[argmin(abs.(k))]
    k1 = k[argmax(abs.(k))]

    # Defining S-matrix
    GA(z) = inv(HV(z) - A'*G0(z)*A)
    GB(z) = inv(HV(z) - B*G0(z)*B')
    UR(z) = [[(1+G0(z)*k1 + G0(z)*(B'*GB(z)*B)[1,1]*G0(z)*k1) (1+G0(z)*k2 + G0(z)*(B'*GB(z)*B)[1,1]*G0(z)*k2)]; [(G0(z)*A*GA(z)*B)[1,1] (G0(z)*A*GA(z)*B)[1,1]]]
    UL(z) = [[G0(z)*(B'*GB(z)*A')[1,1] G0(z)*(B'*GB(z)*A')[1,1]]; [(1+G0(z)/k1 + G0(z)*(A*GA(z)*A')[1,1]*G0(z)/k1) (1+G0(z)/k2 + G0(z)*(A*GA(z)*A')[1,1]*G0(z)/k2)]]
    T = inv(UR(z))*UL(z)
    T[2,2]^(-1)
end