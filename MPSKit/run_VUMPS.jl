using MPSKit, MPSKitModels, TensorKit
using Printf: @sprintf
# using LinearAlgebra: BLAS
include("hamiltonian.jl")

let 

    initial_time = time()

#  -- Physical parameter setup ---
    Ny = 6
    N_sites = 2Ny 
    Jnn, Jnnn, DMI, h = 1, 0.1, 0.0, 0.1
    ani = 0.

    O1, O2 = "S+", "S-"
    operators = "+-"

    dynamic = false
    
    obc_y = true 
    Oy = "t"

#  -- Temperature --
    beta = 1
    # !!! FT 3 threads
    # !!! GS 2 threads

#  -- numerical setup ---
    dmrg_sw = 10
    dmrg_linkdim = 256
    dmrg_maxdim = ones(Int, dmrg_sw) * dmrg_linkdim
    maxdim = 10
    psidim = 256

    N_bands = 4

#  -- evolution accuracy --
    psi_cutoff = 1E-10
    time_cutoff = 1E-10
    println("State cutoff: ", psi_cutoff)

# --- T=0 ---

#  -- physical states --

    # V_phys = U1Space(1 => 1, -1 => 1) # Sz conservation
    # V_phys = ℂ^2

#   - empirical guess of sector windows -
    χ = 10
    # V_virt = ℂ^4
    # V_virt = U1Space(Dict(c => χ for c in 0:N_sites))
    # V_virt = U1Space(-2 => 4, -1 => 4, 0 => 8, 1 => 4, 2 => 4)  # χ≈24, Sz=0 sector

    V_phys_shifted, V_virt = make_shifted_spaces(2Ny, χ, 1)

    # ψ₀ = InfiniteMPS(V_phys, V_virt)
    ψ₀ = InfiniteMPS(
        fill(V_phys_shifted, 2Ny),   # physical spaces
        fill(V_virt, 2Ny)            # virtual spaces (uniform now!)
    )

    mps_time = time()
    println("Time for MPS initialization: ", mps_time - initial_time, " seconds")

#  -- excitation ansatz --
    H = H_HC_uMPS(Ny, Jnn, Jnnn, DMI, h, ani, V_phys_shifted; anc=false, obc_y=obc_y)

    ψ, envs, _ = find_groundstate(ψ₀, H, VUMPS(; tol=psi_cutoff, maxiter=dmrg_sw))

    E₀ = sum(expectation_value(ψ, H))
    println("Ground state energy:  E₀ = $E₀")
    println("Energy per site:      E₀/N = $(E₀ / N_sites)")
    gs_time = time()
    println("Time for ground state search: ", gs_time - mps_time, " seconds")

    # ── Build all shifted environments ───────────────────────────────
    # For a unit cell of size N_sites, need N_sites different insertion points
    ψ_shifts = [circshift(ψ, k) for k in 0:N_sites-1]
    envs_shifts = [environments(ψ_shifts[k+1], H) for k in 0:N_sites-1]


    # ── Excitation spectrum ───────────────────────────────────────────
    alg = QuasiparticleAnsatz(; tol=1e-8)
    momenta = range(-π, π; length=11)

    # sector = trivial for charge-neutral (e.g. magnon) excitations
    # For U1 symmetry, a magnon flips one spin: ΔSz = +1 → U1Irrep(2)
    magnon_sector = U1Irrep(2)   # in 2*Sz units

    # Pass all shifted pairs: (left_ψ, left_env, right_ψ, right_env)
    # MPSKit internally sums over all insertion points
    E_excitations, ϕ_excitations = excitations(
        H, alg, momenta,
        ψ_shifts[1], envs_shifts[1],   # AB ordering
        ψ_shifts[2], envs_shifts[2];   # BA ordering (circshift by 1)
        sector = magnon_sector,
        num = N_bands    # number of bands
    )

    excitation_time = time()
    println("Time for excitation calculation: ", excitation_time - gs_time, " seconds")

    # ── Plot dispersion ───────────────────────────────────────────────
    using Plots
    plot(momenta ./ π, real.(E_excitations),
        xlabel = "k / π",
        ylabel = "E - E₀",
        label  = ["band $i" for i in 1:size(E_excitations,2)],
        title  = "Honeycomb magnon spectrum"
    )

    # --- Save data ---
    filename = @sprintf("VUMPS_HC_Ny%d_Jnn%.2f_Jnnn%.2f_DMI%.2f_h%.2f_ani%.2f_Oy%s_QNt2.csv",
        Ny, Jnn, Jnnn, DMI, h, ani, Oy
    )
    @info "Saving data to $filename"
    open(filename, "w") do io
        write(io, "k")
        for i in 1:N_bands
            write(io, ",E_band_$i")
        end
        write(io, "\n")
        for i in 1:length(momenta)
            k = momenta[i]
            write(io, @sprintf("%.4f", k))
            for j in 1:N_bands
                write(io, @sprintf(",%.6f", real(E_excitations[i,j])))
            end
            write(io, "\n")
        end
    end

    println("Total runtime: ", time() - initial_time, " seconds")

end
