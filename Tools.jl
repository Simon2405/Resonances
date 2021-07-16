# author : Simon Ruget
# This file contains various tools about visualization and studying more precisely those plots (poles finder)

#using Plots
using PyPlot
using Printf
using Polynomials
using PyCall
using LinearAlgebra
using QuadGK
using DSP
using Conda
#Conda.add("scipy")
using RootsAndPoles

#################
### Constants ###
#################
lambdas = range(-5, 5, length=200)
etas = range(-3.5, 1, length=200)


#############################
### Plot Complex Function ###
#############################
function plot_complex_function(λs, ηs, f)
    # Plot levels of the absolute value of f for every z with Re(z) \in λs and Im(z) \in ηs

    cm = range(HSL(0.0, 1.0, 0.5), stop=HSL(360.0, 1.0, 0.5), length=600)
    cm = convert(Array{RGB{Float64}}, cm)
    res = f isa Function ? [f(λ + im*η) for λ in λs, η in ηs] : f

    ## Using Plots
    #p1 = Plots.contour(λs, ηs, log10.(abs.(res))')
    #p2 = Plots.heatmap(λs, ηs, angle.(res)', c=cgrad(cm))
    #Plots.plot(p1)

    ## Using PyPlot
    PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize="x-small")
    PyPlot.rc("ytick", labelsize="x-small")
    PyPlot.rc("figure", figsize=(4,3))
    PyPlot.rc("text", usetex=false)
    f = PyPlot.figure()
    xlabel("real axis")
    ylabel("imaginary axis")
    PyPlot.contour(λs, ηs, abs.(res)', levels = exp.(log(10)*[-0.9, -0.8, -0.7, -0.6, -0.4, -0.2, 0., 0.35]), )#log10.(abs.(res))', levels=[-0.9, -0.8, -0.7, -0.6, -0.4, -0.2, 0., 0.7])
    PyPlot.colorbar()
    #PyPlot.contourf(λs, ηs, log10.(abs.(res))', levels=[-0.9, -0.8, -0.7, -0.6, -0.4, -0.2, 0., 0.7])
    PyPlot.tight_layout()
end




####################
### Poles Finder ###
####################

function finding_poles(S, xb, xe, yb, ye; r=0.01)
    # Search and return poles of the function S in the area [xb, xe] + im*[yb, ye]
    #xb = -4 # real part begin
    #xe = -3  # real part end
    #yb = -0.1  # imag part begin
    #ye = 0.1  # imag part end
    #r = 0.01  # initial mesh step
    tolerance = 1e-9
    
    # Meshing the complex plan
    origcoords = rectangulardomain(complex(xb, yb), complex(xe, ye), r)

    # Finding Poles of S
    zroots, zpoles = grpf(S, origcoords)

    return zpoles
end


