# author : Simon Ruget
# We study here a simple model of 1D diatomic chain where the unit cell consists of one atoms with sites energy p1 
# and one atom with energy p2, each atom is linked cith its nearest nerighbor with energy V

using LinearAlgebra
#using Plots
using PyPlot
using ForwardDiff


#############
### Bands ###
#############
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

function HamilBloch(k; V=1, p1=1, p2=2)
    H_k = [[p1 2*V*cos(k/2)*exp(-im*k/2)]; [2*V*cos(k/2)*exp(im*k/2) p2]]
    return H_k
end


function Band1(k; V=1, p1=1, p2=2)
    return (p1+p2)/2 + sqrt(((p1-p2)^2)/4 + 4*(V^2)*(cos(k/2))^2)
end

function gradBand1(k; V=1, p1=1, p2=2)
    #return (4*(V^2) * (-sin(k/2)) * cos(k/2))/(2 * sqrt(((p1-p2)^2)/4 + 4*(V^2)*(cos(k/2))^2))
    return ForwardDiff.derivative(l->Band1(l; V=V, p1=p1, p2=p2), k)
end


function Band2(k; V=1, p1=1, p2=2)
    return (p1+p2)/2 - sqrt(((p1-p2)^2)/4 + 4*(V^2)*(cos(k/2))^2)
end

function gradBand2(k; V=1, p1=1, p2=2)
    #return -(4*(V^2) * (-sin(k/2)) * cos(k/2))/(2 * sqrt(((p1-p2)^2)/4 + 4*(V^2)*cos(k/2)^2))
    return ForwardDiff.derivative(l->Band1(l; V=V, p1=p1, p2=p2), k)
end




function Bands(; V=1, p1=1, p2=2)
    k = range(-pi, pi, length=100)
    band1 = zeros(100)
    band2 = zeros(100)
    for i in 1:100
        band1[i] = Band1(k[i]; V=V, p1=p1, p2=p2)
        band2[i] = Band2(k[i]; V=V, p1=p1, p2=p2)
    end
    #plot(k, [band1 band2], title="Band relation for diatomic chain, p1=$p1, p2=$p2")
    PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize="x-small")
    PyPlot.rc("ytick", labelsize="x-small")
    PyPlot.rc("figure", figsize=(4,3))
    PyPlot.rc("text", usetex=false)

    f = PyPlot.figure()
    f.suptitle("Dispersion Relation")
    xlabel("k")
    ylabel("Energy")
    PyPlot.plot(k, band1)
    PyPlot.plot(k, band2)
    PyPlot.legend(("Band 1", "Band 2",), loc="upper right")
    PyPlot.tight_layout()
end

function exactBands(; V=1, p1=1, p2=2)
    k = range(-pi, pi, length=100)
    band1 = zeros(100)
    band2 = zeros(100)
    for i in 1:100
        band1[i] = eigen(HamilBloch(k[i]; V=V, p1=p1, p2=p2)).values[1]
        band2[i] = eigen(HamilBloch(k[i]; V=V, p1=p1, p2=p2)).values[2]
    end
    #plot(k, [band1 band2], title="Band relation for diatomic chain, p1=$p1, p2=$p2")
    PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize="x-small")
    PyPlot.rc("ytick", labelsize="x-small")
    PyPlot.rc("figure", figsize=(4,3))
    PyPlot.rc("text", usetex=false)

    f = PyPlot.figure()
    f.suptitle("Dispersion Relation")
    xlabel("k")
    ylabel("Energy")
    PyPlot.plot(k, band1)
    PyPlot.plot(k, band2)
    PyPlot.legend(("Band 1", "Band 2",), loc="upper right")
    PyPlot.tight_layout()
end


##########
### ki ###
##########
function ki_func(k, z, E1, E2, method; a = ((p1+p2)/2)^2, V=1, p1=1, p2=2)
    ## Contour deformation : k -> k + im*ki_func(k, Args)
    ki = 0
    bandnbr = abs( Band1(k; V=V, p1=p1, p2=p2) - real(z) )<=abs( Band2(k; V=V, p1=p1, p2=p2) - real(z) ) ? 1 : 2
    if method==1
        ## ki = -E1 * sign * chi
        if bandnbr==1
            ki = -E1 * sign(gradBand1(k; V=V, p1=p1, p2=p2)) * chi( (Band1(k; V=V, p1=p1, p2=p2)-real(z)) / E2 )
        elseif bandnbr==2
            ki = E1 * sign(gradBand2(k; V=V, p1=p1, p2=p2)) * chi( (Band2(k; V=V, p1=p1, p2=p2)-real(z)) / E2 ) #please note that the sign is not the same as with Band1 case
        end
    
    elseif method==2
        ## ki = -E1 * 1/(a + gradBand^2) * chi
        if bandnbr==1
            ki = -E1 * gradBand1(k; V=V, p1=p1, p2=p2)/(a + gradBand1(k; V=V, p1=p1, p2=p2)^2) * chi( (Band1(k; V=V, p1=p1, p2=p2)-real(z)) / E2 )
        elseif bandnbr==2
            ki = -E1 * gradBand2(k; V=V, p1=p1, p2=p2)/(a + gradBand2(k; V=V, p1=p1, p2=p2)^2) * chi( (Band2(k; V=V, p1=p1, p2=p2)-real(z)) / E2 )
        end
    
    elseif method==3
        ## ki = -E1 * gradBand/(E2 + gradBand^2)
        if bandnbr==1
            ki = -E1 * gradBand1(k; V=V, p1=p1, p2=p2)/(E2 + gradBand1(k; V=V, p1=p1, p2=p2)^2)
        elseif bandnbr==2
            ki = -E1 * gradBand2(k; V=V, p1=p1, p2=p2)/(E2 + gradBand2(k; V=V, p1=p1, p2=p2)^2)
        end
    
    elseif method==4
        ## ki = -E1 * gradBand
        if bandnbr==1
            ki = -E1 * gradBand1(k; V=V, p1=p1, p2=p2)
        elseif bandnbr==2
            ki = -E1 * gradBand2(k; V=V, p1=p1, p2=p2)
        end

    end
    return ki
end

function plot_ki(z, E1, E2, N, method; a = ((p1+p2)/2)^2, V=1, p1=1, p2=2)
    M=100
    k = range(-pi,pi, length=M)
    y=zeros(M)
    for i in 1:M
        y[i] = ki_func(k[i], z, E1, E2, method; a = a, V=V, p1=p1, p2=p2)
    end
    PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize="x-small")
    PyPlot.rc("ytick", labelsize="x-small")
    PyPlot.rc("figure", figsize=(4,3))
    PyPlot.rc("text", usetex=false)

    f = PyPlot.figure()
    f.suptitle("ki_(k)")
    xlabel("k")
    ylabel("ki")
    PyPlot.plot(k, y)
    PyPlot.tight_layout()
end

function gradki_func(k, z, E1, E2, method; a = ((p1+p2)/2)^2, V=1, p1=1, p2=2)
    return ForwardDiff.derivative(l -> ki_func(l, z, E1, E2, method; a=a, V=V, p1=p1, p2=p2), k)
end



#######################
### G0 coefficients ###
#######################
function diatomicmatrixelemG0(m, n, z, E1, E2, N, method; a = ((p1+p2)/2)^2, V=1, p1=1, p2=2)
    ## Compute the m, n'th element matrix of G0  applying contour deformation at energy z.
    # Sampling of the Brillouin Zone
    k = range(-pi, pi, length=N+1)[1:N]

    # Defining Contour deformation with ki
    ki = zeros(N)
    gradki = zeros(N)
    for j in 1:N
        ki[j] = ki_func(k[j], z, E1, E2, method; a=a, V=V, p1=p1, p2=p2)
        gradki[j] = gradki_func(k[j], z, E1, E2, method; a=a, V=V, p1=p1, p2=p2)
    end

    # Defining G0
    G0 = zeros(Complex, 2, 2)
    for j in 1:N
        G0 += exp(im*(k[j] + im*ki[j])*(m-n)) * inv(z*I - HamilBloch(k[j] + im*ki[j]; V=V, p1=p1, p2=p2)) * (1+im*gradki[j])/2
        #G0[1, 1] += exp(im*(k[j] + im*ki[j])*(m-n)) * (1+im*gradki[j]) / (z - Band1(k[j] + im*ki[j]))
        #G0[2, 2] += exp(im*(k[j] + im*ki[j])*(m-n)) * (1+im*gradki[j]) / (z - Band2(k[j] + im*ki[j]))
    end
    G0 = G0/N
    return G0
end



#################
### Integrand ###
#################
function integrand_diatomic(z, E1, E2, N, method; a=((p1+p2)/2)^2 +1)
    ## Plot the integrand of the integral defining the element matrix of G0
    k = range(-pi, pi, length = N+1)[1:N]
    ki = zeros(N)
    gradki = zeros(N)
    for j in 1:N
        ki[j] = ki_func(k[j], z, E1, E2, method; a)
        gradki[j] = gradki_func(k[j], z, E1, E2, method; a)
    end 

    integr = zeros(Complex, N)
    for j in 1:N
        integr[j] += (1+im*gradki[j]) / (z - Band1(k[j] + im*ki[j]))
        integr[j] += (1+im*gradki[j]) / (z - Band2(k[j] + im*ki[j]))
    end

    PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize="x-small")
    PyPlot.rc("ytick", labelsize="x-small")
    PyPlot.rc("figure", figsize=(4,3))
    PyPlot.rc("text", usetex=false)

    f = PyPlot.figure()
    f.suptitle("integrand, method=$method, E1=$E1, E2=$E2, a=$a")
    xlabel("k")
    ylabel("integrand")
    PyPlot.plot(k, imag.(integr))
    PyPlot.tight_layout()

    #plot(k[1:N], imag.(integr), title="integrand, E1 = $E1, E2 = $E2, N = $N")
    #plot(k, [integr[1,:] integr[2,:] integr[3,:] integr[4,:]], title = "E1=$E1, E2 = $E2, N=$N")
end

###########
### DOS ###
###########
function DOSdiatomic1D(E)
    ## Compute DOS with naiv method (using a big finite system and a small eta)
    eta = 0.01
    N=1000
    
    # Hamiltonian diatomic
    H = zeros((N,N))
    for i in 1:N
        if (i%2)==1
            H[i,i] = p1
        else
            H[i,i] = p2
        end
    end
    for i in 1:(N-1)
        H[i, i+1] = V
        H[i+1, i] = V
    end

    return -imag( tr(inv( (E+im*eta)*I - H ) ) )/(N*pi)
end


function DOSdiatomic1D_Contour(E1,E2, N, method; a=((p1+p2)/2)^2 +1, V=1, p1=1, p2=2)
    ## Plot the DOS using contour deformation and using naiv method
    M=100
    x = range((p1+p2)/2 - sqrt((p1-p2)^2/4 + 4*V^2) -0.5, (p1+p2)/2 + sqrt((p1-p2)^2/4 + 4*V^2) +0.5, length=M)
    y = zeros(M)
    for i in 1:M
        y[i] = -imag(tr(diatomicmatrixelemG0(1, 1, x[i], E1, E2, N, method;a=a, V=V, p1=p1, p2=p2)))/pi
    end

    y_exact = DOSdiatomic1D.(x)
    
    #plot(x, [y y_exact], label=["DOS" "DOS_exact"], title="method=$method, E1=$E1, E2=$E2, N=$N")
    PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize="x-small")
    PyPlot.rc("ytick", labelsize="x-small")
    PyPlot.rc("figure", figsize=(4,3))
    PyPlot.rc("text", usetex=false)

    f = PyPlot.figure()
    f.suptitle("method=$method, E1=$E1, E2=$E2, N_BZ=$N")
    xlabel("Energy")
    ylabel("DOS")
    PyPlot.plot(x, y)
    PyPlot.plot(x, y_exact)
    PyPlot.legend(("DOS contour", "DOS naive",), loc="upper right")
    PyPlot.tight_layout()
end