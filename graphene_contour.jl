# author : Simon Ruget
# This files contains function to compute resonances due to an addatoms on 2D-graphene


using LinearAlgebra
using Plots
#using PyPlot
using ForwardDiff


#################
### Constants ###
#################

V = 2.6 # jump constant
eps = 2*V # coupling term
E = 1.7 #energy of initial state

W = zeros((3, 3)) # Perturbation (the defect is an addatoms eps-coupled with the first atom of (0,0) unit-cell
W[1, 2] = eps
W[2, 1] = eps

ψ0 = zeros(3) # Initial state is purely localized in the defect
ψ0[1] = 1


# Graphene Mesh
r = 1 #distance between atoms
delta1 = (r/2) * [1 sqrt(3)] #1st nearest neighbors
delta2 = (r/2) * [1 -sqrt(3)] #2nd nearest neighbors
delta3 = r * [-1 0] # 3rd nearest neighbors
b1 = (2*pi/(3*r)) * [1 sqrt(3)] # Brillouin Zone
b2 = (2*pi/(3*r)) * [1 -sqrt(3)]
Ms = 0.



#############
### Bands ###
#############

function Hamiltonian_C(N)
    V = 2.6 
    ## Create the hamiltonian of the graphene mesh with N atoms
    H = zeros(2, N^2, 2, N^2)
    for i in 1:(N-1)
        for j in 1:(N-1)
            H[1, i+N*(j-1), 2, i+1+N*(j)] = V
            H[2, i+1+N*(j), 1, i+N*(j-1)] = V
            H[1, i+N*(j-1), 2, i+1+N*(j-1)] = V
            H[2, i+1+N*(j-1), 1, i+N*(j-1)] = V
            H[1, i+N*(j-1), 2, i+N*(j)] = V
            H[2, i+N*(j), 1, i+N*(j-1)] = V
        end
    end
    H = reshape(H, (2*N^2, 2*N^2))
    return H
end

function H_C(k)
    # Graphene Parameters
    r = 1. #distance between atoms
    delta1 = (r/2) * [1 sqrt(3)] #1st nearest neighbors
    delta2 = (r/2) * [1 -sqrt(3)] #2nd nearest neighbors
    delta3 = r * [-1 0] # 3rd nearest neighbors
    V = 2.6 # jump constant
    Ms = 0.

    # Computing Bloch Hamiltonian
    H = zeros(Complex{Float64}, 2, 2)
    D = exp(im * dot(k, delta1)) + exp(im * dot(k, delta2)) + exp(im * dot(k, delta3))
    H[1, 2] = -V * D
    H[2, 1] = -V * D'
    H[1, 1] = -V*Ms
    H[2, 2] = V*Ms
    return H
end

function Bands1_C(k)
    # Graphene Parameters 
    r =1. #distance between atoms
    V = 2.6 # jump constant

    return V*sqrt(1 + 4 * cos(3*r*k[1]/2) * cos(sqrt(3)*r*k[2]/2) + 4 * (cos(sqrt(3)*r*k[2]/2))^2)
end

function gradBands1_C(k)
    return ForwardDiff.gradient(Bands1_C, k)
end


function Bands2_C(k)
    # Graphene Parameters 
    r =1. #distance between atoms
    V = 2.6 # jump constant
    return -V*sqrt(1 + 4 * cos(3*r*k[1]/2) * cos(sqrt(3)*r*k[2]/2) + 4 * (cos(sqrt(3)*r*k[2]/2))^2)
end

function gradBands2_C(k)
    return ForwardDiff.gradient(Bands2_C, k)
end


function Bands_C()
    # Graphene Parameters
    r = 1. #distance between atoms
    b1 = (2*pi/(3*r)) * [1 sqrt(3)] # Brillouin Zone
    b2 = (2*pi/(3*r)) * [1 -sqrt(3)]

    # Show Bands
    N = 100
    x = range(-dot(b1, [1 0]), dot(b1, [1 0]), length=N)
    y = range(dot(b2, [0 1]), dot(b1, [0 1]), length= N)
    B1(x, y) = Bands1_C([x y])
    B2(x, y) = Bands2_C([x y])
    res1 = zeros(Float64, N, N)
    res2 = zeros(Float64, N, N)
    for i in 1:N
        for j in 1:N
            res1[i, j] = B1(x[i], y[j])
            res2[i, j] = B2(x[i], y[j])
        end
    end

    #=PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize="x-small")
    PyPlot.rc("ytick", labelsize="x-small")
    PyPlot.rc("figure", figsize=(4,3))
    PyPlot.rc("text", usetex=false)
    
    f = PyPlot.figure()
    f.suptitle("Bands of Graphene")
    PyPlot.xlabel("kx")
    PyPlot.ylabel("ky")
    PyPlot.zlabel("E(kx, ky)")
    PyPlot.plot_surface(x, y, res1)
    PyPlot.plot_surface(x, y, res2)
    PyPlot.tight_layout()=#

    Plots.plot(x, y, B2, st=:surface, camera=(20,50))
    Plots.plot!(x, y, B1, st=:surface, camera=(20,50), xlabel = "kx", ylabel = "ky", zlabel="Bands of Graphene")
end


##########
### ki ###
##########

function chi(x)
    ## Cut Off Function
    return(exp(-(x^2)/2))
end

function gradchi(x)
    ## Derivative of Cut Off Function 
    return(-x*chi(x))
end

function ki_C(k::AbstractVector{T}, Band, gradBand, z, E1, E2, method; a = 3*V) where T
    ki = zeros(T, 2)
    r = 1. #distance between atoms
    V = 2.6 # jump constant
    if method==1
        ## ki = -E1 * sign * chi
        ki = .- E1 .* sign.(gradBand(k)) .* chi((Band(k)-real(z)) /E2)
    
    elseif method==2
        ## ki = -E1 * gradBand/(a + gradBand^2) * chi
        ki = -E1 * gradBand(k)/(a + norm(gradBand(k))^2) * chi( (Band(k)-real(z)) / E2 )
    
    elseif method==3
        ## ki = -E1 * gradBand/(E2 + gradBand^2)
        ki = -E1 * gradBand(k)/(E2 + norm(gradBand(k))^2)
    
    elseif method==4
        ## ki = -E1 * gradBand,
        ki = -E1 * gradBand(k)
    
    elseif method==5
        ## ki = -E1 * gradBand/(a + gradBand) * chi
        ki = -E1 * gradBand(k)/(a + norm(gradBand(k))) * chi( (Band(k)-real(z)) / E2 )

    elseif method==6
        ## ki = -E1 * gradBand *chi( (Band(k)-real(z)) / E2 )
        ki[1] = -E1 * chi( (Band(k)-real(z)) / E2 ) * 4 * V * cos(sqrt(3)*k[2]/2)*(-sin(3*k[1]/2))
        ki[2] = -E1 * chi( (Band(k)-real(z)) / E2 ) * 4 * V *( sqrt(3) * cos(3*k[1]/2)*(-sin(sqrt(3)*k[2]/2))/2 + sqrt(3)*(-sin(sqrt(3)*k[2]/2)) * cos(sqrt(3)*k[2]/2) )
    end
    return ki
end


function jacobianki_C(k::AbstractVector{T}, Band, gradBand, z, E1, E2, method; a = 3*V) where T
    return I+im*ForwardDiff.jacobian(l->ki_C(l, Band, gradBand, z, E1, E2, method; a), k)
end






#######################
### G0 coefficients ###
#######################
function matrixelement_G0_C(m, n, z, E1, E2, N, method; a = 3*V)
    ## Compute the (m,n)'th matrix element of G0 for the graphene at energy z
    # Graphene Parameters
    r = 1. #distance between atoms
    b1 = (2*pi/(3*r)) * [1, sqrt(3)] # Brillouin Zone
    b2 = (2*pi/(3*r)) * [1, -sqrt(3)]
    E = 5. #energy of initial state

    ## This function returns the (m,n) matrix-element G0(z)[m,n] calculated with the contour integral
    # Sampling of the Brillouin Zone
    u = range(0, 1, length=N+1)[1:N]
    v = range(0, 1, length=N+1)[1:N]

    # Defining Contour deformation with ki
    ki = zeros(Float64, N, N, 2)
    detJac = zeros(Complex{Float64}, N, N)
    for i in 1:N
        for j in 1:N
            k = u[i]*b1 + v[j]*b2 #[u[i]*b1[1]+v[j]*b2[1], u[i]*b1[2]+v[j]*b2[2]]
            bandnbr = abs( Bands1_C(k) - real(z) )<=abs( Bands2_C(k) - real(z) ) ? 1 : 2
            if bandnbr==1
                ki[i,j, :] = ki_C(k, Bands1_C, gradBands1_C, z, E1, E2, method; a)
                detJac[i, j] = det(jacobianki_C(k, Bands1_C, gradBands1_C, z, E1, E2, method; a))
            else
                ki[i,j, :] = ki_C(k, Bands2_C, gradBands2_C, z, E1, E2, method; a)
                detJac[i, j] = det(jacobianki_C(k, Bands2_C, gradBands2_C, z, E1, E2, method; a))
            end
        end
    end

    # Defining G0
    G0 = zeros(Complex{Float64}, 2, 2)
    for i in 1:N
        for j in 1:N
            kprime = u[i]*b1 + v[j]*b2 + im*ki[i,j,:] #[u[i]*b1[1]+v[j]*b2[1], u[i]*b1[2]+v[j]*b2[2]] + im*ki[i,j,:] #cannot use u[i]*b1 + v[j]*b2 cause type issue (2d row vector vs 1d column vector)
            G0 += cis(dot(kprime, m-n)) * detJac[i, j] * inv(z*I - H_C(kprime))
        end
    end
    aire = norm(b2)*dot(b1, [sqrt(3)/2, 1/2])/(N*N) # normalization factor of G0
    G0 = G0*aire

    return G0
end


###########################
### Integrand and plots ###
###########################
function integrand_2D_C(z, E1, E2, N, method; a = 3*V)
    ## Plot the integrand as a heatmap
    # Graphene Parameters
    r = 1. #distance between atoms
    b1 = (2*pi/(3*r)) * [1 sqrt(3)] # Brillouin Zone
    b2 = (2*pi/(3*r)) * [1 -sqrt(3)]

    ## This function plots an heatmap of the 2D integrand that we use in our contour integrals
    # Sampling of the Brillouin Zone
    x = range(-dot(b1, [1 0]), dot(b1, [1 0]), length=N)
    y = range(dot(b2, [0 1]), dot(b1, [0, 1]), length= N)

    # Defining Contour deformation with ki
    ki = zeros(Float64, N, N, 2)
    detJac = zeros(Complex{Float64}, N, N)
    for i in 1:N
        for j in 1:N
            k = [x[i], y[j]]
            bandnbr = abs( Bands1_C(k) - real(z) )<=abs( Bands2_C(k) - real(z) ) ? 1 : 2
            if bandnbr==1
                ki[i,j, :] = ki_C(k, Bands1_C, gradBands1_C, z, E1, E2, method; a)
                detJac[i, j] = det(jacobianki_C(k, Bands1_C, gradBands1_C, z, E1, E2, method; a))
            else
                ki[i,j, :] = ki_C(k, Bands2_C, gradBands2_C, z, E1, E2, method; a)
                detJac[i, j] = det(jacobianki_C(k, Bands2_C, gradBands2_C, z, E1, E2, method; a))
            end
        end
    end
    
    # Defining integrand
    res = zeros(Float64, N, N)
    for i in 1:N
        for j in 1:N
            kprime = [x[i], y[j]] + im*ki[i,j,:]
            res[i, j]= imag(tr(detJac[i, j] * inv(z*I - H_C(kprime))))/(pi)
        end
    end
    #=PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize="x-small")
    PyPlot.rc("ytick", labelsize="x-small")
    PyPlot.rc("figure", figsize=(4,3))
    PyPlot.rc("text", usetex=false)
    
    f = PyPlot.figure("")
    f.suptitle("z=z, E1=E1, E2=E2, N=$N, method=$method; a=$a")
    PyPlot.xlabel("kx")
    PyPlot.ylabel("ky")
    PyPlot.zlabel("integrand 2D")
    PyPlot.imshow(res)
    PyPlot.colorbar()
    PyPlot.tight_layout()=#

    heatmap(x, y, res, title="E1=$E1, E2=$E2, a=$a, N=$N", xlabel = "kx", ylabel = "ky", zlabel="integrand2D")
end


function quiver_ki_C(z, E1, E2, N, method; a = 10)
    ## Quiver plot of ki
    # Graphene Parameters
    r = 1. #distance between atoms
    b1 = (2*pi/(3*r)) * [1 sqrt(3)] # Brillouin Zone
    b2 = (2*pi/(3*r)) * [1 -sqrt(3)]
    B = [b2' b1']

    ## This function plots an heatmap of the absolute value of ki
    # Sampling of the Brillouin Zone
    x = repeat(range(0, 1, length=N), 1, N) |> vec
    y = repeat(range(0, 1, length=N), 1, N)' |> vec

    # Plotting the applied deformation ki
    k(x,y) = [x*b2[1]+y*b1[1], x*b2[2]+y*b1[2]]
    ki1(x,y) = abs( Bands1_C(k(x,y)) - real(z) )<=abs( Bands2_C(k(x,y)) - real(z) ) ? ki_C(k(x,y), Bands1_C, gradBands1_C, z, E1, E2, method; a)[1] : ki_C(k(x,y), Bands2_C, gradBands2_C, z, E1, E2, method; a)[1]
    ki2(x,y) = abs( Bands1_C(k(x,y)) - real(z) )<=abs( Bands2_C(k(x,y)) - real(z) ) ? ki_C(k(x,y), Bands1_C, gradBands1_C, z, E1, E2, method; a)[2] : ki_C(k(x,y), Bands2_C, gradBands2_C, z, E1, E2, method; a)[2]
    vx = ki1.(x,y) |> vec
    vy = ki2.(x,y) |> vec

    Plots.quiver(x, y, quiver=(vx, vy), title="Contour deformation (blue) and Fermi surface (red) at energy E=$z", xlabel = "kx", ylabel = "ky", aspect_ratio=:equal)
    #=PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize="x-small")
    PyPlot.rc("ytick", labelsize="x-small")
    PyPlot.rc("figure", figsize=(4,3))
    PyPlot.rc("text", usetex=false)
    
    f = PyPlot.figure()
    f.suptitle("k_i(k)")
    xlabel("kx")
    ylabel("ky")
    PyPlot.quiver(x, y, vx, vy, scale=1.8, scale_units="xy", units="xy")=#
    # Plotting the Fermi surface
    x=range(0, 1, length=N)
    y=range(0, 1, length=N)
    Z =zeros(N, N)
    for i in 1:N
        for j in 1:N
            Z[i, j] = (abs( Bands1_C(B*[x[i] y[j]]') - real(z) )<=abs( Bands2_C(B*[x[i] y[j]]') - real(z) ) ? 1 : -1)*2.6*sqrt(1 + 4 * cos(3*(B*[x[i] y[j]]')[1]/2) * cos(sqrt(3)*(B*[x[i] y[j]]')[2]/2) + 4 * (cos(sqrt(3)*(B*[x[i] y[j]]')[2]/2))^2)
        end
    end
    contour!(x,y, (x,y)->V(x,y), levels=[z], aspect_ratio=:equal)
    #=PyPlot.contour(x,y,Z, levels=[z], colors=:red)
    PyPlot.tight_layout()=#
end

function plot_ki(z, E1, E2, N, method; a = 3*V)
    ## Plot ki as a heatmap
    # Graphene Parameters
    r = 1. #distance between atoms
    b1 = (2*pi/(3*r)) * [1 sqrt(3)] # Brillouin Zone
    b2 = (2*pi/(3*r)) * [1 -sqrt(3)]

    ## This function plots an heatmap of the absolute value of ki
    # Sampling of the Brillouin Zone
    x = range(-dot(b1, [1 0]), dot(b1, [1 0]), length=N)
    y = range(dot(b2, [0 1]), dot(b1, [0, 1]), length= N)

    # Defining Contour deformation with ki
    ki = zeros(Float64, N, N)
    for i in 1:N
        for j in 1:N
            k = [x[i], y[j]]
            bandnbr = abs( Bands1_C(k) - real(z) )<=abs( Bands2_C(k) - real(z) ) ? 1 : 2
            if bandnbr==1
                ki[i,j] = norm(ki_C(k, Bands1_C, gradBands1_C, z, E1, E2, method; a))
            else
                ki[i,j] = norm(ki_C(k, Bands2_C, gradBands2_C, z, E1, E2, method; a))
            end
        end
    end
    heatmap(x, y, ki, title="E1=$E1, E2=$E2, N=$N, a=$a", xlabel = "kx", ylabel = "ky")
end



function plot_jacki(z, E1, E2, N, method; a = 3*V)
    # Graphene Parameters
    r = 1. #distance between atoms
    b1 = (2*pi/(3*r)) * [1 sqrt(3)] # Brillouin Zone
    b2 = (2*pi/(3*r)) * [1 -sqrt(3)]

    ## This function plots an heatmap of the absolute value of det(Jacobian_ki)
    # Sampling of the Brillouin Zone
    x = range(-dot(b1, [1 0]), dot(b1, [1 0]), length=N)
    y = range(dot(b2, [0 1]), dot(b1, [0, 1]), length=N)

    # Defining Contour deformation with ki
    jacki = zeros(Float64, N,  N)
    for i in 1:N
        for j in 1:N
            k = [x[i], y[j]]
            bandnbr = abs( Bands1_C(k) - real(z) )<=abs( Bands2_C(k) - real(z) ) ? 1 : 2
            if bandnbr==1
                jacki[i,j] = abs(det(jacobianki_C(k, Bands1_C, gradBands1_C, z, E1, E2, method; a)))
            else
                jacki[i,j] = abs(det(jacobianki_C(k, Bands2_C, gradBands2_C, z, E1, E2, method; a)))
            end
        end
    end
    heatmap(x, y, jacki, title="E1=$E1, E2=$E2, N=$N", xlabel="kx", ylabel="ky")
end

function plot_gif(E1, E2, N, method, pl; a=3*V)
    # Graphene Parameters
    r = 1. #distance between atoms
    b1 = (2*pi/(3*r)) * [1 sqrt(3)] # Brillouin Zone
    b2 = (2*pi/(3*r)) * [1 -sqrt(3)]

    ## This function creates a gif that plots the integrand (pl=1), the jacobian of ki (pl=2),
    ## the ki (pl=3), z-Band (pl=4) for various z.
    M=100
    z_list = range(-3*V +0.5, 3*V -0.5, length=M)
    x = range(-dot(b1, [1 0]), dot(b1, [1 0]), length=N)
    y = range(dot(b2, [0 1]), dot(b1, [0, 1]), length=N)
    
    function I(z)
        # Defining Contour deformation with ki
        ki = zeros(Float64, N, N, 2)
        detJac = zeros(Complex{Float64}, N, N)
        for i in 1:N
            for j in 1:N
                k = [x[i], y[j]]
                bandnbr = abs( Bands1_C(k) - real(z) )<=abs( Bands2_C(k) - real(z) ) ? 1 : 2
                if bandnbr==1
                    ki[i,j, :] = ki_C(k, Bands1_C, gradBands1_C, z, E1, E2, method; a)
                    detJac[i, j] = det(jacobianki_C(k, Bands1_C, gradBands1_C, z, E1, E2, method; a))#-[[1 0]; [0 1]])
                else
                    ki[i,j, :] = ki_C(k, Bands2_C, gradBands2_C, z, E1, E2, method; a)
                    detJac[i, j] = det(jacobianki_C(k, Bands2_C, gradBands2_C, z, E1, E2, method; a))#-[[1 0]; [0 1]])
                end
            end
        end

        # Defining plot
        res = zeros(N, N)
        for i in 1:N
            for j in 1:N
                if pl == 1 ## plot integrand
                    kprime = [x[i], y[j]] + im*ki[i,j,:]
                    res[i,j] = imag(tr(detJac[i, j] * inv(z*[[1 0]; [0 1]] - H_C(kprime))))/pi
                elseif pl == 2 ## plot det(jacobian)
                    res[i,j] = abs(detJac[i,j])
                elseif pl == 3 ## plot ki
                    res[i,j] = norm(ki[i,j,:])
                elseif pl == 4 ## plot z-Band_k
                    kprime = [x[i], y[j]] + im*ki[i,j,:]
                    res[i,j] = abs( Bands1_C([x[i], y[j]]) - real(z) )<=abs( Bands2_C([x[i], y[j]]) - real(z) ) ? real(z-Bands1_C(kprime)) : real(z-Bands2_C(kprime))
                end
            end
        end
        return res
    end
    
    function E(z)
        ## Plot the Bands B1, B2 and the constant surface B3(x,y) = z
        L = 100
        x_L = range(-dot(b1, [1 0]), dot(b1, [1 0]), length=L)
        y_L = range(dot(b2, [0 1]), dot(b1, [0, 1]), length=L)
        B1(x, y) = Bands1_C([x y])
        B2(x, y) = Bands2_C([x y])
        B3(x, y) = z
        Plots.plot(x_L, y_L, B2, st=:surface, camera=(45,10))
        Plots.plot!(x_L, y_L, B1, st=:surface, camera=(45,10))
        Plots.plot!(x_L, y_L, B3, st=:surface, camera=(45,10))
    end

    anim = @animate for i in 1:M
        e = z_list[i]
        res = I(e)
        figure1=heatmap(x, y, res, title="$e", clim=(-1, 1))
        figure2=E(e)
        Plots.plot(figure1, figure2)
    end
    gif(anim, fps = 1)
end;



###########
### DOS ###
###########

function DOSExact_C(E)
    # Graphene Parameters
    r = 1. #distance between atoms
    b1 = (2*pi/(3*r)) * [1 sqrt(3)] # Brillouin Zone
    b2 = (2*pi/(3*r)) * [1 -sqrt(3)]

    ## This function calculates the Density of State at energy E using the formula DOS(E) = -Imag(Tr(G0(E+im*eta)))/pi
    eta = 0.1
    N = 100
    u = range(0, 1, length=N)
    v = range(0, 1, length=N)
    int = 0.
    for i in 1:N
        for j in 1:N
            int += 1/(E + im*eta - Bands1_C([u[i]*b1[1]+v[j]*b2[1], u[i]*b1[2]+v[j]*b2[2]]))
            int += 1/(E + im*eta - Bands2_C([u[i]*b1[1]+v[j]*b2[1], u[i]*b1[2]+v[j]*b2[2]]))
        end
    end
    int = -imag(int)/(N*N*pi)
    return int
end

function DOS_Contour_C(E1,E2, N, method; a=3*V)
    ## This function plot the DOS with two methods : using the contour deformation 
    ## and using the formula DOS(E) = -Imag(Tr(G0(E+im*eta)))/pi
    M=100
    x = range(-3*V -0.5, 3*V +0.5, length=M)
    y = zeros(Float64, M)
    for i in 1:M
        if method==0
            y[i] = -imag(tr(matrixelement_G0_C([0., 0.], [0., 0.], x[i]+im*0.1, E1, E2, N, method; a)))/pi
        else
            y[i] = -imag(tr(matrixelement_G0_C([0., 0.], [0., 0.], x[i], E1, E2, N, method; a)))/pi #to test coefficient try with x[i]+im0.1 and method =0
        end
    end

    y_exact = DOSExact_C.(x)
    PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize="x-small")
    PyPlot.rc("ytick", labelsize="x-small")
    PyPlot.rc("figure", figsize=(4,3))
    PyPlot.rc("text", usetex=false)
    
    f = PyPlot.figure()
    f.suptitle("method=$method, E1=$E1, E2=$E2, N=$N, a=$a")
    PyPlot.xlabel("Energy")
    PyPlot.ylabel("DOS")
    PyPlot.plot(x, y)
    PyPlot.plot(x, y_exact)
    PyPlot.legend(("DOS contour", "DOS eta"))
    PyPlot.tight_layout()

    ##Plots.plot(x, [y y_exact], label=["DOS" "DOS Exacte"])
end

#########################
### Fermi Golden Rule ###
#########################
function FGR_C(E, eps)
    ## This function calculates the imaginary parts of the resonances 
    ## according to Fermi Golden Rules when starting with an initial state (of energy E) in the defect
    # Graphene Parameters
    r = 1. #distance between atoms
    b1 = (2*pi/(3*r)) * [1 sqrt(3)] # Brillouin Zone
    b2 = (2*pi/(3*r)) * [1 -sqrt(3)]
    
    # FGR
    eta = 0.1
    N = 100
    u = range(0, 1, length=N)
    v = range(0, 1, length=N)
    int = 0
    for i in 1:N
        for j in 1:N
            int += dot([1 0], inv((E + im*eta)I - H_C([u[i]*b1[1]+v[j]*b2[1], u[i]*b1[2]+v[j]*b2[2]]))*[1 0]')
        end
    end
    int = -imag(int)/(N*N*pi)
    return pi * eps^2 * int
end
