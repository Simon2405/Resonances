# author : Simon Ruget
# Contains various function about the implementation of the Green Function additives approach

using LinearAlgebra
using Plots
using ForwardDiff
#using Images
#using ImageView
#using NonlinearEigenproblems

#################
### Constants ###
#################
eps = 0.25 # coupling
V = 2 # jump constant in the defect
W = zeros((6, 6)) # Perturbation
W[1, 2] = eps-1
W[2, 3] = V-1
W[3, 4] = V-1
W[4, 5] = V-1
W[5, 6] = eps-1
W[2, 1] = eps-1
W[3, 2] = V-1
W[4, 3] = V-1
W[5, 4] = V-1
W[6, 5] = eps-1

#Defining initial state
HV = [[0 V 0 0]; [V 0 V 0]; [0 V 0 V]; [0 0 V 0]]
lambdaV, VV = eigen(HV)
ψ0 = zeros(6)
ψ0[2:5] = VV[:,3]
lambdas=range(0.8, 1.6, length=100)
etas=range(-0.1, 0.1, length=100)

#Contour deformation
E1 = 0.2
E2 = 0.6
N=20


########################
### Cut Off Function ###
########################
function chi(x)
    #if abs(x) > 1
    #    return 0
    #end
    #return exp(-1/(1-x^2))*exp(1)
    return(exp(-x^2/2))
end

function gradchi(x)
    #if abs(x)> 1
    #    return 0
    #end
    #return -2*exp(1)*x*exp(-1/(1-x^2))/(1-x^2)^2
    return(-x*chi(x))
end

##########
### ki ###
##########
function ki_func_1D(k, z, E1, E2, method; a = 2)
    ki = 0
    if method==1
        ki = -E1 * sign(-2*sin(k)) * chi( (2*cos(k)-real(z)) / E2 )
    
    elseif method==2
        ki = -E1 * (-2*sin(k))/(a + (-2*sin(k))^2) * chi( (2*cos(k)-real(z)) / E2 )
    
    elseif method==3
        ki = -E1 * (-2*sin(k))/(E2 + (-2*sin(k))^2)
    
    elseif method==4
        ki = -E1 * (-2*sin(k))

    end
    return ki
end

function gradki_func_1D(k, z, E1, E2, method; a = 2)
    return ForwardDiff.derivative(l -> ki_func_1D(l, z, E1, E2, method; a), k)
end

###########################################################################################
######################################### 1D Case #########################################
###########################################################################################


#######################
### G0 coefficients ###
#######################

function matrixelemG0(m,n,z, method;E1=0.5, E2=0.5, N=500, a = 2, E= 1.23)
    ### This function compute the (m,n)'th coefficients of the unperturbed Green function
    # Meshing the Brillouin zone ([-∏, ∏])
    k = range(-pi, pi, length=N+1)
    
    # Calculating imaginary part of the contour deformation kstar(k) = k + iki(k) and gradki
    ki = zeros(N)
    gradki = zeros(N)
    for j in 1:N
        ki[j] = ki_func_1D(k[j], E, E1, E2, method; a)
        gradki[j] = gradki_func_1D(k[j], E, E1, E2, method; a)
    end

    
    # Calculating the matrix Element G0[m,n]
    MatrixElem = 0
    for j in 1:N
        MatrixElem += exp(im*(k[j] + im*ki[j])*(m-n)) * (1 + im*gradki[j]) / (z-2*cos(k[j] + im*ki[j]))
    end
    MatrixElem = MatrixElem/N

    return(MatrixElem)
end

###########
### DOS ###
###########
function DOS(E)
    ## Using a small parameters \eta, return -imag.(Tr(G0(E + im*eta)))/pi
    eta = 0.01
    N=1000
    k=range(-pi, stop=pi, length = N)
    int = 0
    for u in 1:N
        int += 1/(E + im*eta - 2*cos(k[u]))
    end
    int = -imag(int)/(N*pi)
    return int
end

function DOS_Contour(E1,E2, N, method;a=0.5)
    ## Plot the DOS using contour deformation metho and using eta method
    x = range(-2.5, 2.5, length=100)
    y = zeros(100)
    for i in 1:100
        y[i] = -imag.(matrixelemG0(1,1,x[i], method;E1=E1, E2=E2, N=N, a=a))/pi
    end

    y_exact = zeros(100)
    for i in 1:100
        y_exact[i] = DOS(x[i])
    end
    plot(x, [y y_exact], label=["DOS" "DOS_exact"], title="E1=$E1, E2=$E2, N=$N")
    #return -imag.(matrixelemG0(1,1,x;E1=E1, E2=E2, N=N, a=a))/pi
end




###########################################################
### Matrix element \langle \psi_0 | G(z) \psi_0 \rangle ###
###########################################################

function AdditiveGreenFunction(z, W, ψ0, E1, E2, N, method)
    Size_defect = size(W)[1]
    
    # G0
    G0 = zeros(Complex, Size_defect, Size_defect)
    for i in 1:Size_defect
        for j in 1:Size_defect
            G0[i,j] = matrixelemG0(i,j,z, method; E1=E1, E2=E2, N=N)
        end
    end
    
    Pψ = inv(I-G0*W)*G0*ψ0
    ψ = (G0*ψ0 + G0*W*Pψ)[:,1]
    return ψ0'*ψ
end

function detAdditiveGreenFunction(z, W, E1, E2, N, method)
    Size_defect = size(W)[1]
    
    # G0
    G0 = zeros(Complex, Size_defect, Size_defect)
    for i in 1:Size_defect
        for j in 1:Size_defect
            G0[i,j] = matrixelemG0(i,j,z,method; E1=E1, E2=E2, N=N)
        end
    end
    
    P = inv(I-G0*W)*G0
    return det(P)
end

##############################
### Encapsulated Functions ###
##############################
function plottingAdditiveGreenFunction(lambdas, etas, W, ψ0, E1, E2, N, method)
    res = [AdditiveGreenFunction(lbd+im*et, W, ψ0, E1, E2, N, method) for lbd in lambdas, et in etas]
end

function plottingdetAdditiveGreenFunction(lambdas, etas, W, E1, E2, N, method)
    res = [detAdditiveGreenFunction(lbd+im*et, W, E1, E2, N, method) for lbd in lambdas, et in etas]
end

function polesfindingAdditiveGreenFunction(z)
    AdditiveGreenFunction(z, W, ψ0, E1, E2, N, method)
end

function polesfindingdetAdditiveGreenFunction(z)
    detAdditiveGreenFunction(z, W, E1, E2, N, method)
end


######################
### Resonant State ###
######################
function ResonantState_naive(z, Size; eps=0.25, V=2)
    ## Naive method to find Resonant State : we just take a big system and consider the closest to 0 eigenvalue
    ## Define H-z operator
    H = zeros(Complex, Size, Size)
    for i in 1:(Size-1)
        H[i, i] = -z
        H[i, i+1] = 1
        H[i+1, i] = 1
    end
    H[Size, Size] = -z

    for i in (Int((Size-6)/2) +2):(Int((Size-6)/2) +4)
        H[i, i+1] = 2
        H[i+1, i] = 2
    end
    H[(Int((Size-6)/2) +1), (Int((Size-6)/2) +2)] = eps
    H[(Int((Size-6)/2) +2), (Int((Size-6)/2) +1)] = eps
    H[(Int((Size-6)/2) +5), (Int((Size-6)/2) +6)] = eps
    H[(Int((Size-6)/2) +6), (Int((Size-6)/2) +5)] = eps

    eigval, eigvect = eigen(H)
    index = argmin(abs.(eigval))
    return eigvect[:, index]
end

function ResonantState(z, W, E1, E2, N, method, M, E)
    ## Resonant State computed with contour deformation method
    # Resize W
    Wr = zeros(2*M+6, 2*M+6)
    Wr[(M+1):(M+6), (M+1):(M+6)] = W

    # G0
    G0 = zeros(Complex, 2*M+6, 2*M+6)
    for i in 1:(2*M+6)
        for j in 1:(2*M+6)
            G0[i,j] = matrixelemG0(i,j,z,method; E1=E1, E2=E2, N=N, E=E)
        end
    end

    # Finding state
    P = zeros(6, 2*M+6)
    P[1:6, (M+1):(M+6)] = I(6)

    projpsi = eigen(I-P*G0*Wr*P').vectors[:,1]
    ψ = (G0*Wr*P'*projpsi)[:,1]

    return ψ

    #Imz = round(imag(z), digits=3)
    #Rez = round(real(z), digits=3)
    #Plots.plot(1:(2*M+6), [real.(ψ) imag.(ψ) abs.(ψ)], label=["real" "imag" "abs"], title="Resonant State of energy $Rez $Imz im", marker=2)
    
end



#################
### Integrand ###
#################
function Plot_contour_deformation(z, E1, E2, N, metho; a = 0.5)
    ## Plot the profile of ki
    k = range(-pi, pi, length = N)
    contour = zeros(Complex, N)
    for j in 1:N
        ki = ki_func_1D(k[j], z, E1, E2, method; a = a)
        contour[j] = exp(im*(k[j] + im*ki))
    end
    plot(real.(contour), imag.(contour), title="E1 = $E1, E2 = $E2, N = $N")
        
end

function integrand(z, E1, E2, N, method; a=0.5)
    ## Plot the integrand of the integral defining G0
    k = range(-pi, pi, length = N)
    integr = zeros(Complex, N)
    for j in 1:N
        ki = ki_func_1D(k[j], z, E1, E2, method; a = a)
        gradki = gradki_func_1D(k[j], z, E1, E2, method; a = a)
        integr[j] = (1+im*gradki)/(z-2*cos(k[j] + im*ki))
    end 
    plot(k, imag.(integr), title="integrand, E1 = $E1, E2 = $E2, N = $N")
    #plot(k, [integr[1,:] integr[2,:] integr[3,:] integr[4,:]], title = "E1=$E1, E2 = $E2, N=$N")
end

###################
### Convergence ###
###################

function Convergence_N(E1, E2, method; expo=false)
    ## Study convergence of the pole : for various size of BZ sampling, we compute the poles
    list_N = 5:65 #[10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50]
    poles = zeros(Complex, size(list_N)[1])
    Golden_Rule = -(eps^2 * sqrt(4-lambdaV[3]^2) * (ψ0[2]^2 + ψ0[5]^2)/2)*ones(size(list_N)[1])
    for i in 1:size(list_N)[1]
        polesfindingAdditiveGreenFunctionN(z) = AdditiveGreenFunction(z, W, ψ0, E1, E2, list_N[i], method)
        p = finding_poles(polesfindingAdditiveGreenFunctionN, 0.8, 1.6, -0.1, 0.1)
        if size(p)[1]>=1
            poles[i] = p[1]
        end
    end
    return list_N, poles
    #figure1 = plot(list_N, [imag.(poles) Golden_Rule], marker = 2, title="Resonances as a function of N (method=$method, E1=$E1, E2=$E2)", label=["Resonances" "Golden Rule"])
    #figure2 = plot(list_N, real.(poles), marker = 2, label ="energy shift")
    #lot(figure1, figure2)
    #convergence = zeros(size(list_N)[1]-1)
    #for i in 1:(size(list_N)[1]-1)
    #    convergence[i] = abs(imag.(poles)[i] - imag.(poles)[size(list_N)[1]])==0 ? 1 : abs(imag.(poles)[i] - imag.(poles)[size(list_N)[1]])
    #end
    #figure3 = plot(list_N[1:(size(list_N)[1]-1)], convergence, yaxis=:log)
    #return poles, convergence
end
