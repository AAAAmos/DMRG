using MPSKit, MPSKitModels, TensorKit
using Printf: @sprintf
# using LinearAlgebra: BLAS
include("hamiltonian.jl")

let 

    initial_time = time()

#  -- Physical parameter setup ---
    N = 3
    M = 3
    N_sites = 2N*M 
    Jnn, Jnnn, DMI, h = 1, 0.1, 0.0, 0.1
    ani = 0.

    O1, O2 = "S+", "S-"
    operators = "+-"

    dynamic = false
    
    obc_x = true  
    obc_y = false 
    Ox, Oy = "t", "f"

#   - obc x, pbc y -
    # col_x = 4
    # centerA, centerB = 1+2*M*(col_x-1), 2+2*M*(col_x-1)

#   - obc y, pbc x -
    row_y = 1
    centerA, centerB = 1+2*(row_y-1), 2+2*(row_y-1)

    # -finite T-

    # centerA *= 2
    # centerB *= 2

#  -- Temperature --
    beta = 1
    # !!! FT 3 threads
    # !!! GS 2 threads

#  -- momentum --
    #=  a_1, a_2 = [√(3), 0], [√(3), 3]/2
        K+: [2, 1]/3
        K-: [1, 2]/3
        M: [1, 1]/2
    =#
    Q_list = []
    Q_text = []
    n_k = 0

#  -- numerical setup ---
    dmrg_sw = 10
    dmrg_linkdim = 256
    dmrg_maxdim = ones(Int, dmrg_sw) * dmrg_linkdim
    maxdim = 10
    psidim = 256

#  -- evolution accuracy --
    psi_cutoff = 1E-10
    time_cutoff = 1E-10
    println("State cutoff: ", psi_cutoff)

#  -- Time step --
    dtau = 0.05
    tausweep = 1
    Tsteps = 400

# --- T=0 ---

#  -- physical states --

    # V_phys = ℂ^2
    V_phys = U1Space(1 => 1, -1 => 1) # Sz conservation

#   - empirical guess of sector windows -
    χ = 4
    V_virt = U1Space(Dict(c => χ for c in 0:N_sites))
    # V_virt = U1Space(-2 => 4, -1 => 4, 0 => 8, 1 => 4, 2 => 4)  # χ≈24, Sz=0 sector

#   - uniform sectors -
    # V_virt = ℂ^4

    # sector_per_dim = 4
    # sector_window = 8
    # V_virt = U1Space(q=>sector_per_dim for q in -sector_window:sector_window)
    # println("Total initial bond dim = $(sector_per_dim*(2sector_per_dim+1))")

    ψ₀ = FiniteMPS(
        N_sites, V_phys, V_virt;
        left  = U1Space(0 => 1),      # left boundary always trivial
        right = U1Space(N_sites => 1)        # right boundary fixes total Sz
    )

#  -- dmrg --
   
    H = H_HC(true, N, M, Jnn, Jnnn, DMI, h, ani; anc=false, obc_x, obc_y)

    alg = DMRG(
        tol      = psi_cutoff,
        maxiter  = dmrg_sw,
        verbosity = 2
    )

    ψ, envs, δ = find_groundstate(ψ₀, H, alg)

#  -- observables --
    N_sites = 2N*M
    # S_z = TensorMap([1/2 0; 0 -1/2], ℂ^2, ℂ^2)

    E₀ = sum(expectation_value(ψ, H))
    println("Ground state energy:  E₀ = $E₀")
    println("Energy per site:      E₀/N = $(E₀ / N_sites)")

    # sz_profile = real.(expectation_value(ψ, [S_z], 1:N_sites))
    # println("\n⟨Sz⟩ profile:")
    # for (i, sz) in enumerate(sz_profile)
    #     println("  site $i : $(round(sz, digits=6))")
    # end

end
