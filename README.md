# Resonances
Various scripts that helps calculating quantum resonances in material physics

Here is a short description of the files :
- Tools.jl contains functions that are useful to plot and compute resonances (using a solver)
- Scattering.jl contains a function that calculate the S-matrix for a specific case : a 1D linear chain with a 4 sites defect in the middle
- Additive_Green_Function.jl contains various function related to the computation of resonances using deformation contour method in the 1D case
- Additive_Green_Function_2D.jl contains various function related to the computation of resonances using deformation contour method in the 2D case
- diatomic_chain.jl contains various function related to the computation of resonances using deformation contour method in the specific 1D case with a two atoms unit cell
- graphene_contour.jl contains various function related to the computation of resonances using deformation contour method for graphene.

You'll need to download the following packages to use this project : 
- PyPlot.jl
- ForwardDiff.jl
- RootsAndPoles.jl
