# author : Simon Ruget
# Contains various function about the implementation of the Green Function additives approach in 2D

using LinearAlgebra
#using Plots
using PyPlot
using ForwardDiff

#################
### Constants ###
#################

eps = 0.1
E = 2 # potential in the detAdditiveGreenFunction2D
method = 2

W = zeros((2, 2)) # Perturbation (the defect is a potential localized in a single site coupled with its 4 nearest neighbors)
W[1, 2] = eps
W[2, 1] = eps

ψ0 = zeros(2) # Initial state is purely localized in the defect
ψ0[1] = 1


###########################################################################################
######################################### 2D Case #########################################
###########################################################################################

##########
### ki ###
##########

function Hamiltonian_2D(N)
    ## Create the finite Hamiltonian for 2D square mesh
    H = zeros(N^2, N^2)
    for i in 2:(N-1)
        for j in 2:(N-1)
            H[i + N*(j-1), i+1 + N*(j-1)] = 1
            H[i+1 + N*(j-1), i + N*(j-1)] = 1

            H[i + N*(j-1), i-1 + N*(j-1)] = 1
            H[i-1 + N*(j-1), i + N*(j-1)] = 1

            H[i + N*(j-1), i + N*(j)] = 1
            H[i + N*(j), i + N*(j-1)] = 1

            H[i + N*(j-1), i + N*(j-2)] = 1
            H[i + N*(j-2), i + N*(j-1)] = 1
        end
    end
    return H
end

function chi(x)
    ## Cut Off Function
    return(exp(-(x^2)/2))
end

function gradchi(x)
    ## Derivative of Cut Off Function 
    return(-x*chi(x))
end

function energy_2D(k)
    ## Relation of dispersion links E with k through the following relation
    return 2*cos(k[1]) + 2*cos(k[2])
end

function gradEnergy_2D(k)
    return ForwardDiff.gradient(energy_2D, k)
end

function plot_energy()
    N=100
    x = range(-pi, pi, length=N)
    y = range(-pi, pi, length=N)
    z = zeros(N, N)
    for i in 1:N
        for j in 1:N
            z[i, j] = energy_2D([x[i], y[j]])
        end
    end

    PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize="x-small")
    PyPlot.rc("ytick", labelsize="x-small")
    PyPlot.rc("figure", figsize=(4,3))
    PyPlot.rc("text", usetex=false)

    f = PyPlot.figure()
    f.suptitle("Disperion Relation")
    PyPlot.xlabel("kx")
    PyPlot.ylabel("ky")
    PyPlot.zlabel("E(kx, ky)")
    PyPlot.plot_surface(x, y, z)
    PyPlot.tight_layout
end


function ki_2D(k::AbstractVector{T}, z, E1, E2, method; a = 1) where T
    ## Contour deformation : k -> k + im*ki_2D(k, args)
    ki = zeros(T, 2)
    if method==1
        ## ki = -E1 * sign * chi
        ki = .- E1 .* sign.(gradEnergy_2D(k)) .* chi((energy_2D(k)-real(z)) /E2)
    
    elseif method==2
        ## ki = -E1 * gradBand/(a + gradBand^2) * chi
        ki = -E1 * gradEnergy_2D(k)/(a + norm(gradEnergy_2D(k))^2) * chi( (energy_2D(k)-real(z)) / E2 )
    
    elseif method==3
        ## ki = -E1 * gradBand/(E2 + gradBand^2)
        ki = -E1 * gradEnergy_2D(k)/(E2 + norm(gradEnergy_2D(k))^2)
    
    elseif method==4
        ## ki = -E1 * gradBand,
        ki = -E1 * gradEnergy_2D(k)
    
    elseif method==5
        ## ki = -E1 * gradBand/(a + gradBand) * chi
        ki = -E1 * gradEnergy_2D(k)/(a + norm(gradEnergy_2D(k))) * chi( (energy_2D(k)-real(z)) / E2 )
    
    elseif method==6
        ## ki = -E1 * sign(grad_Ek) * ksi((E_k - Re(z))/E2)
        ki[1] = -E1 * chi((energy_2D(k)-real(z))/E2) * (-sin(k[1]))
        ki[2] = -E1 * chi((energy_2D(k)-real(z))/E2) * (-sin(k[2]))
    end
    return ki
end


function jacobianki_2D(k::AbstractVector{T}, z, E1, E2, method; a = 1) where T
    return I+im*ForwardDiff.jacobian(l->ki_2D(l, z, E1, E2, method; a), k)
end

#######################
### G0 Coefficients ###
#######################

function matrixelemG02D(m,n,z, method; E1 = 0.1, E2=0.5, N=200, a=1)
    # Parameters
    E = 2.

    # Meshing the Brillouin zone ([-∏, ∏]*[-∏,∏])
    kx = range(-pi, pi, length=N+1)
    ky = range(-pi, pi, length=N+1)
    
    
    # Calculating k_i
    ki = zeros((N,N,2))
    gradki = zeros(Complex, N, N)
    for u in 1:N
        for v in 1:N
            ki[u, v, :] = ki_2D([kx[u], ky[v]], E, E1, E2, method; a) 
            gradki[u, v] = det(jacobianki_2D([kx[u], ky[v]], E, E1, E2, method; a) )
        end
    end

    # Matrix Element
    MatrixElem = 0
    for u in 1:N
        for v in 1:N
            h = (kx[u]+im*ki[u,v,1])*(m[1]-n[1]) + (ky[v]+im*ki[u,v,2])*(m[2]-n[2])
            MatrixElem += cis(im*h) * gradki[u,v] / (z - 2*cos(kx[u] + im*ki[u,v,1]) - 2*cos(ky[v] + im*ki[u,v,2]))
        end
    end
    MatrixElem = MatrixElem/(N*N)
    return MatrixElem
end

#############
### Plots ###
#############

function integrand2D(z, E1, E2, N, method; a=1)
    # Meshing the Brillouin zone ([-∏, ∏]*[-∏,∏])
    kx = range(-pi, pi, length=N+1)
    ky = range(-pi, pi, length=N+1)

    # Calculating k_i
    ki = zeros((N,N,2))
    gradki = zeros(Complex, N, N)
    for u in 1:N
        for v in 1:N
            ki[u, v, :] = ki_2D([kx[u], ky[v]], z, E1, E2, method; a) 
            gradki[u, v] = det(jacobianki_2D([kx[u], ky[v]], z, E1, E2, method; a) )
        end
    end
    
    res = zeros(N,N)
    for u in 1:N
        for v in 1:N
            res[u,v] = -imag(gradki[u,v] / (z - 2*cos(kx[u] + im*ki[u,v,1]) - 2*cos(ky[v] + im*ki[u,v,2])))
        end
    end
    PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize="x-small")
    PyPlot.rc("ytick", labelsize="x-small")
    PyPlot.rc("figure", figsize=(4,3))
    PyPlot.rc("text", usetex=false)

    f = PyPlot.figure()
    f.suptitle("z=$z, method=$method, E1=$E1, E2=$E2, N=$N")
    xlabel("kx")
    ylabel("ky")
    zlabel("integrand2D")
    PyPlot.plot_surface(kx[1:N], ky[1:N], res)
    PyPlot.tight_layout()
    #contour(real.(exp.(kx)))
    #return(kx[11], ky[1], res[11, 1])
end

function quiver_ki(z, E1, E2, N, method; a = 1)
    ## This function plots an heatmap of the absolute value of ki
    # Sampling of the Brillouin Zone
    x = repeat(range(-pi, pi, length=N), 1, N) |> vec
    y = repeat(range(-pi, pi, length=N), 1, N)' |> vec

    # Plotting the applied deformation ki
    ki1(x,y) = ki_2D([x, y], z, E1, E2, method; a)[1]
    ki2(x,y) = ki_2D([x, y], z, E1, E2, method; a)[2]
    vx = ki1.(x,y) |> vec
    vy = ki2.(x,y) |> vec
    ##Plots.quiver(x, y, quiver=(vx, vy), title="Contour deformation (blue) and Fermi surface (red) at energy E=$z", xlabel = "kx", ylabel = "ky", aspect_ratio=:equal)
    PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize="x-small")
    PyPlot.rc("ytick", labelsize="x-small")
    PyPlot.rc("figure", figsize=(4,3))
    PyPlot.rc("text", usetex=false)
    
    f = PyPlot.figure()
    f.suptitle("k_i(k)")
    xlabel("kx")
    ylabel("ky")
    PyPlot.quiver(x, y, vx, vy, scale=1.8, scale_units="xy", units="xy")
    
    # Plotting the Fermi surface
    x=range(-pi, pi, length=N)
    y=range(-pi, pi, length=N)
    Z = zeros(N,N)
    for i in 1:N
        for j in 1:N
            Z[i,j] = energy_2D([x[i], y[j]])
        end
    end
    ##Plots.contour!(x,y, (x,y)->energy_2D([x, y]), levels=[z], aspect_ratio=:equal, c=:red)
    PyPlot.contour(x,y,Z, levels=[z], colors=:red)
    PyPlot.tight_layout()

end



function G(N, eta, eps, site, M; E1 = 0.4, E2= 0.6, a=1, method = 6, N_BZ=20)
    ## Plot G(z)=(d|1/(z+imxeta-H)|d) with H describing perturbed system
    energy= range(-4.5, 4.5, length=M)
    E = 2 
    
    ## Method with large finite system
    Hamiltonian = zeros(N^2+1, N^2+1)
    Hamiltonian[1, 1] = E
    Hamiltonian[1, Int(N/2) + N*(Int(N/2)-1)] = eps
    Hamiltonian[Int(N/2) + N*(Int(N/2)-1), 1] = eps
    Hamiltonian[2:(N^2+1), 2:(N^2+1)] = Hamiltonian_2D(N)
    eigval, eigvect = eigen(Hamiltonian)
    
    ψ0 = zeros(N^2 + 1)
    if site == -1
        ψ0[1] = 1
        ψ0_index = 1
    else # system's symmetry allows us to just think of the sites along only one axis (we choosed x bellow)
        ψ0[Int(N/2) + site + N*(Int(N/2)-1)] = 1
        ψ0_index = Int(N/2) + site + N*(Int(N/2)-1)
    end


    y_1 = zeros(M)
    for i in 1:M
        for eigindex in 1:size(eigval)[1] ## Sums over eigencouples
            y_1[i] += -imag(abs(eigvect[ψ0_index, eigindex])^2/(energy[i]+im*eta-eigval[eigindex]))
        end
    end
    atomsnbr = N^2
    ##Plots.plot(energy, y_1, title="G(z)=(d|1/(z+imxeta-H)|d), eta=$eta, eps=$eps, d=$site", label="finite large sys, sys_size = $atomsnbr", legend=:bottomleft, xlabel="energy")
    PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize="x-small")
    PyPlot.rc("ytick", labelsize="x-small")
    PyPlot.rc("figure", figsize=(4,3))
    PyPlot.rc("text", usetex=false)

    f = PyPlot.figure()
    f.suptitle("Green function on adatom")
    xlabel("Energy")
    ylabel("G(z)=(d|1/(z+imxeta-H)|d)")
    PyPlot.plot(energy, y_1)
    PyPlot.tight_layout()
end

function plot_deformation(λs, ηs, f, N, E1, E2, method; a = 1)
    plot_complex_function(λs, ηs, f)
    
    # Poles
    kx = range(-pi, pi, length=N+1)[1:N]
    ky = range(-pi, pi, length=N+1)[1:N]
    E=2
    # Calculating k_i
    ki = zeros((N,N,2))
    for u in 1:N
        for v in 1:N
            ki[u, v, :] = ki_2D([kx[u], ky[v]], E, E1, E2, method; a)
        end
    end
    
    # Calculating poles due to contour deformation
    poles = zeros(Complex, N*N)
    for u in 1:N
        for v in 1:N
            poles[u + N*(v-1)] = energy_2D([kx[u], ky[v]]+im*ki[u, v, :])
        end
    end

    PyPlot.scatter(real.(poles), imag.(poles), marker="+", c=:black)
    PyPlot.tight_layout()
end
#############################################################
### Matrix element \langle \psi_0 | G(z) | \psi_0 \rangle ###
#############################################################

function AdditiveGreenFunction2D(z, E1, E2, N, method)
    # W
    W=zeros(2,2)
    eps = 0.1
    W[1, 2] = eps
    W[2, 1] = eps
    ψ0 = zeros(2)
    ψ0[1] = 1
    E=2

    # G0
    G0 = zeros(Complex, 2,2)
    G0[1,1] = 1/(z-E)
    G0[2,2] = matrixelemG02D([0 0],[0 0],z, method; E1=E1, E2=E2, N=N)
    
    Pψ = inv(I-G0*W)*G0*ψ0
    ψ = (G0*ψ0 + G0*W*Pψ)[:,1]
    return ψ0'*ψ
end

function detAdditiveGreenFunction2D(z, E1, E2, N, method)
    # W
    W=zeros(2,2)
    eps = 1.5
    W[1, 2] = eps
    W[2, 1] = eps
    ψ0 = zeros(2)
    ψ0[1] = 1
    E=2

    # G0
    G0 = zeros(Complex, 2,2)
    G0[1,1] = 1/(z-E)
    G0[2,2] = matrixelemG02D([0 0],[0 0],z, method; E1=E1, E2=E2, N=N)

    P = inv(I-G0*W)
    return det(P)
end

function AdditiveGreenFunction2D_prime(z, E1, E2, N, method)
    #=W=zeros(10,10)
    eps = 0.1
    W[1, 6] = eps
    W[2, 6] = eps
    ψ0 = zeros(10)
    ψ0[1] = 1

    
    # G0
    G0 = zeros(Complex, 10,10)
    G0[1,1] = 1/(z-E)

    for i in 2:10
        for j in 2:10
            m = [(i-1)%3 Int((i-1-(i-1)%3)/3 + 1)]
            n = [(j-1)%3 Int((j-1-(j-1)%3)/3 + 1)]
            G0[i,j] = matrixelemG02D(m,n,z, method; E1=E1, E2=E2, N=N)
        end
    end=#
    W=zeros(2,2)
    eps = 1.5
    W[1, 2] = eps
    W[2, 1] = eps
    ψ0 = zeros(2)
    ψ0[1] = 1
    E=2

    
    # G0
    G0 = zeros(Complex, 2,2)
    G0[1,1] = 1/(z-E)
    G0[2,2] = matrixelemG02D([0 0], [0 0], z, method; E1=E1, E2=E2, N=N)

    #=Pψ = inv(I-G0*W)*G0*ψ0
    ψ = (G0*ψ0 + G0*W*Pψ)[:,1]
    return ψ0'*ψ=#
    P = inv(I - G0*W)*G0
    return det(P)
end

function operatorG0V_2D(z, E1, E2, N, method)
    W=zeros(2,2)
    eps = 1.
    W[1, 2] = eps
    W[2, 1] = eps
    ψ0 = zeros(2)
    ψ0[1] = 1
    E=2

    
    # G0
    G0 = zeros(Complex, 2,2)
    G0[1,1] = 1/(z-E)
    G0[2,2] = matrixelemG02D([0 0], [0 0], z, method; E1=E1, E2=E2, N=N)

    #=Pψ = inv(I-G0*W)*G0*ψ0
    ψ = (G0*ψ0 + G0*W*Pψ)[:,1]
    return ψ0'*ψ=#
    P = G0*W
    return det(P)
end

#############################
### Encapsulated Function ###
#############################
function plottingAdditiveGreenFunction2D(lambdas, etas, E1, E2, N, method)
    res = [AdditiveGreenFunction2D(lbd+im*et, E1, E2, N, method) for lbd in lambdas, et in etas]
end

function plottingdetAdditiveGreenFunction2D(lambdas, etas, E1, E2, N, method)
    res = [detAdditiveGreenFunction2D(lbd+im*et, E1, E2, N, method) for lbd in lambdas, et in etas]
end

function plottingAdditiveGreenFunction2D_prime(lambdas, etas, E1, E2, N, method)
    res = [AdditiveGreenFunction2D_prime(lbd+im*et, E1, E2, N, method) for lbd in lambdas, et in etas]
end

function plottingOperatorG0V_2D(lambdas, etas, E1, E2, N, method)
    res = [operatorG0V_2D(lbd+im*et, E1, E2, N, method) for lbd in lambdas, et in etas]
end

function polesfindingdetAdditiveGreenFunction2D(z)
    detAdditiveGreenFunction2D(z, 0.9, 1.4, 20, 6)
end

function polesfindingAdditiveGreenFunction2D(z)
    AdditiveGreenFunction2D(z, 0.6, 1., 20, 2)
end

function polesfindingAdditiveGreenFunction2D_prime(z)
    AdditiveGreenFunction2D_prime(z, 0.9, 1.4, 20, 6)
end


###########
### DOS ###
###########
function DOS2D(E)
    ## Using a small parameters \eta, plots -imag.(Tr(G0(E + im*eta)))/pi
    eta = 0.01
    N=1000
    kx=range(-pi, stop=pi, length = N+1)
    ky = range(-pi, stop=pi, length = N+1)
    int = 0
    for u in 1:N
        for v in 1:N
            int += 1/(E + im*eta - 2*cos(kx[u])-2*cos(ky[v]))
        end
    end
    int = -imag(int)/(N*N*pi)
    return int
end

function DOS2D_Contour(E1,E2,N, method)
    ## Plot the DOS using contour deformation method and using naiv mthod with eta
    M=100
    x = range(-4.5, 4.5, length=M)
    y = zeros(M)
    for i in 1:M
        y[i] = -imag.(matrixelemG02D([1 1]',[1 1]',x[i], method; E1=E1, E2=E2, N=N))/pi
    end
    y_exact = zeros(M)
    for i in 1:M
        y_exact[i] = DOS2D(x[i])
    end
    ##Plots.plot(x, [y y_exact], label=["DOS" "DOS_exact"], title="DOS2D, E1=$E1, a=$E2, N=$N")
    f = PyPlot.figure()
    f.suptitle("Approximation of 2D DOS with N=$N")
    xlabel("Energy")
    ylabel("DOS")
    PyPlot.plot(x, y)
    PyPlot.plot(x, y_exact)
    PyPlot.legend(("Approx E1=$E1, E2=$E2, method=$method", "Exact eta=1.5"), loc="upper left")
    PyPlot.tight_layout()
end


###################
### Convergence ###
###################
function Convergence2D_N(E1, E2, method; expo=false)
    E= 2
    eps=1.5

    list_N = [10, 12, 14, 16, 18, 20, 22, 24]
    poles = zeros(Complex, size(list_N)[1])

    Golden_Rule = - pi * eps^2 * DOS2D(E) * ones(size(list_N)[1])

    for i in 1:size(list_N)[1]
        polesfindingAdditiveGreenFunctionN(z) = AdditiveGreenFunction2D_prime(z, E1, E2, list_N[i], method)
        p = finding_poles(polesfindingAdditiveGreenFunctionN, 1, 3, -1, 0.1)
        if size(p)[1]>=1
            amin = argmin(abs.(imag.(p) - 0.7246073942184442*ones(size(p)[1])))
            poles[i] = p[amin]
        end
    end
    if expo==false #Only plot the convergence of imaginary and real part of poles
        #figure1 = Plots.plot(list_N, [imag.(poles) Golden_Rule], marker = 2, title="Resonances as a function of N (method=$method, E1=$E1, E2=$E2)", label=["Resonances" "Golden Rule"], legend=:bottomright)
        #figure2 = Plots.plot(list_N, real.(poles), marker = 2, label ="energy shift")
        #Plots.plot(figure1, figure2)
        f = PyPlot.figure()
        f.suptitle("Resonances as a function of N (E1=$E1, E2=$E2)")
        xlabel("N: Brillouin-Discretization Parameter")
        ylabel("Imag Part")
        PyPlot.plot(list_N, imag.(poles), marker="o")
        PyPlot.plot(list_N, Golden_Rule)
        PyPlot.legend(("Contour Integral", "Fermi Golden Rule"))
    else #Exponential convergence ?
        convergence = zeros(size(list_N)[1]-1)
        for i in 1:(size(list_N)[1]-1)
            convergence[i] = abs(poles[i] - poles[size(list_N)[1]])
        end

        # Linear Regression
        X = zeros(size(list_N)[1]-1, 2)
        X[:, 1] = ones(size(list_N)[1]-1)'
        X[:, 2] = list_N[1:(size(list_N)[1]-1)]'
        beta = inv(X'*X)*X'*log.(convergence)

        reg = zeros(size(list_N)[1]-1)
        for i in 1:(size(list_N)[1]-1)
            reg[i] = exp(beta[1]+beta[2]*list_N[i])
        end
        #figure = plot(list_N[1:(size(list_N)[1]-1)], [convergence reg], yaxis=:log, marker=2, title = "Convergence", label=["absolute error" "linear regression"], xlabel="Brillouin-Discretization parameter")
        f = PyPlot.figure()
        f.suptitle("Absolute Error showing exponential convergence")
        xlabel("N: Brillouin-Discretization Parameter")
        ylabel("Absolute Error")
        yscale("log")
        PyPlot.plot(list_N[1:(size(list_N)[1]-1)], convergence, marker="o")
        PyPlot.plot(list_N[1:(size(list_N)[1]-1)], reg, marker="o")
        PyPlot.legend(("Absolute Error", "Linear Regression"), loc="upper right")
    end
end

######################
### Resonant State ###
######################

function ResonantState_2D(z, W, E1, E2, N, method, M)
    ## Finding Resonant State with contour deformation method
    Imz = round(imag(z), digits=3)
    Rez = round(real(z), digits=3)


    # Resize W
    Wr = zeros(2*M+6, 2*M+6)
    Wr[M:(M+2), M:(M+5)] = W

    # G0
    G0 = zeros(Complex, 2*M+6, 2*M+6)
    for i in 1:(2*M+6)
        for j in 1:(2*M+6)
            G0[i,j] = matrixelemG0(i,j,z,method; E1=E1, E2=E2, N=N)
        end
    end

    # Finding state
    P = zeros(6, 2*M+6)
    P[1:6, M:(M+5)] = I(6)

    #return abs(det(I-P*G0*Wr*P'))
    projpsi = eigen(I-P*G0*Wr*P').vectors[:,1]
    ψ = (G0*Wr*P'*projpsi)[:,1]


    Plots.plot(1:(2*M+6), [real.(ψ) imag.(ψ) abs.(ψ)], label=["real" "imag" "abs"], title="Resonant State of energy $Rez $Imz im", marker=2)
    
    ## Checking if we the relation G0*(z-H0) is verified
    #H0 = zeros(Complex, 2*M+6, 2*M+6)
    #for i in 1:(2*M+5)
    #    H0[i, i+1] = 1
    #    H0[i+1, i] = 1
    #end#

    #Mat= G0*(z*I-H0)
    #Matabs = zeros(2*M+6, 2*M+6)
    #for i in 1:(2*M+6)
    #    for j in 1:(2*M+6)
    #        Matabs[i,j] = abs(Mat[i,j])
    #    end
    #end

    #imshow(Matabs)
end