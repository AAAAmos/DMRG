using MPSKit, MPSKitModels, TensorKit
using Printf: @sprintf
using LinearAlgebra: svdvals, BLAS
using JLD2
include("iDMRG.jl")

let 

    initial_time = time()
    # BLAS.set_num_threads(1)

#  -- Physical parameter setup ---
    Ny = 4
    N_sites = 2Ny 
    Jnn, Jnnn, DMI, h = 1, 0.1, 0.1, 1.1
    ani = 0.

    obc_y = true 
    Oy = "t"

#  -- numerical setup ---
    dmrg_sw = 200

    N_bands = 1

#  -- evolution accuracy --
    psi_cutoff = 1E-9
    println("State cutoff: ", psi_cutoff)

# --- T=0 ---

#  -- physical states --

    # V_phys = U1Space(1 => 1, -1 => 1) # Sz conservation
    # V_phys_shifted = ℂ^2

#   - empirical guess of sector windows -
    χ = 12
    # V_virt = ℂ^χ
    # V_virt = U1Space(Dict(c => χ for c in 0:N_sites))
    # V_virt = U1Space(-2 => 4, -1 => 4, 0 => 8, 1 => 4, 2 => 4)  # χ≈24, Sz=0 sector

    V_phys_shifted, V_virt, N_qnsectors = make_shifted_spaces(2Ny, χ, -1)

    # Start from a product state (χ=1 per U1 sector) so the initial state is
    # deterministic and subspace expansion adds directions guided by H.
    _, V_virt_init, _ = make_shifted_spaces(2Ny, 1, -1)
    ψ = InfiniteMPS(
        fill(V_phys_shifted, N_sites),
        fill(V_virt_init, N_sites)
    )

    mps_time = time()
    println("Time for MPS initialization: ", mps_time - initial_time, " seconds")
    println("Initial bond dims  : ", bond_dim_str(ψ))

#  -- excitation ansatz --
    H = H_HC_uMPS(Ny, Jnn, Jnnn, DMI, h, ani, V_phys_shifted; anc=false, obc_y=obc_y)

    # Print max bond dim every 20 VUMPS iterations via the finalize callback.
    vumps_mon = (iter, ψ, H, envs) -> begin
        iter % 20 == 0 && println(
            @sprintf("  [VUMPS %4d] max bond dim = %d",
                iter, maximum(dim(right_virtualspace(ψ, n)) for n in 1:length(ψ)))
        )
        return ψ, envs
    end

#  -- Subspace expansion: grow bond dim in stages for reproducible convergence --
    # OptimalExpand (Zauner-Stauber 2018): projects H*AC2 onto the nullspaces of
    # AL and AR, then SVD-truncates to find the best new virtual directions.
    vumps_warm  = VUMPS(; tol=1e-7,       maxiter=100,      finalize=vumps_mon)
    vumps_final = VUMPS(; tol=psi_cutoff, maxiter=dmrg_sw,  finalize=vumps_mon)

    ψ, envs, _ = find_groundstate(ψ, H, vumps_warm)
    println("  χ=1 converged: ", bond_dim_str(ψ))

    # Incremental expansion: each step adds (χ_i - χ_prev)*N_qnsectors new directions.
    χ_prev = 1
    # for χ_i in unique([max(2, χ ÷ 2), χ])
    for χ_i in unique([χ])
        n_add = (χ_i - χ_prev) * N_qnsectors
        χ_prev = χ_i
        println("  Expanding +$n_add states toward χ=$χ_i/sector...")
        ψ, envs = changebonds(ψ, H, OptimalExpand(; trscheme=truncrank(n_add)), envs)
        # ψ, envs = changebonds(ψ, H, OptimalExpand(; trscheme=trunctol(; atol=1e-60)), envs)
        ψ, envs, _ = find_groundstate(ψ, H, vumps_warm, envs)
        println("  Bond dims: ", bond_dim_str(ψ))
    end

    ψ, envs, _ = find_groundstate(ψ, H, vumps_final, envs)

    E₀ = sum(expectation_value(ψ, H))
    # println("Ground state energy:  E₀ = $E₀")
    println("Energy per site:      E₀/N = $(E₀ / N_sites)")
    gs_time = time()
    println("Time for ground state search: ", gs_time - mps_time, " seconds")
    println("Ground state bond dims: ", bond_dim_str(ψ))

    V_gs         = right_virtualspace(ψ, 1)
    qn_sectors_gs = collect(sectors(V_gs))
    χ_gs          = [dim(V_gs, c) for c in qn_sectors_gs]
    total_bd      = dim(V_gs)
    println("Per-sector bond dims: ", join(["$(c.charge)→$(d)" for (c,d) in zip(qn_sectors_gs, χ_gs)], "  "))

    # ── Gauge-fix: make every C[n] diagonal so B tensors from QuasiparticleAnsatz
    #    are computed in a canonical gauge (reproducible entanglement spectra).
    ψ    = gauge_fix_mps(ψ)
    envs = environments(ψ, H)

    # ── Build all shifted environments ───────────────────────────────
    # For a unit cell of size N_sites, need N_sites different insertion points
    ψ_shifts = [circshift(ψ, k) for k in 0:N_sites-1]
    envs_shifts = [environments(ψ_shifts[k+1], H) for k in 0:N_sites-1]


    # ── Excitation spectrum ───────────────────────────────────────────
    alg = QuasiparticleAnsatz(; tol=1e-8)
    # momenta = range(-π/Ny, π/Ny; length=6)
    momenta = range(0, π/Ny; length=6)
    # momenta = range(0.01, 1.2; length=40)

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
    save_object("./uMPS_exci_N$Ny.jld2", ϕ_excitations)

    excitation_time = time()
    println("Time for excitation calculation: ", excitation_time - gs_time, " seconds")

#  -- entanglement --
    println("\nComputing excitation entanglement spectra…")
    entangle_time = time()

    SchmidtCoef = [
        [svdvals(excitation_entanglement_center(ϕ_excitations[ik, b]))
         for b in 1:N_bands]
        for ik in eachindex(momenta)
    ]

    # ϵ_i = -2 log SchmidtCoef_i
    # [ik => χ*N_qnsectors]
    entanglement_spectra = Dict(c => -2.0 .* log.(v[1]) for (c, v) in pairs(SchmidtCoef))

    println("Time for entanglement spectra: ", time() - entangle_time, " seconds")

# --- Save data ---
    filename = @sprintf("VUMPS_HC_Ny%d_Jnn%.2f_Jnnn%.2f_DMI%.2f_h%.2f_ani%.2f_Oy%s_QNt_dim%d_fix1.csv",
        Ny, Jnn, Jnnn, DMI, h, ani, Oy, total_bd
    )
    @info "Saving data to $filename"
    open(filename, "w") do io
        # headers
        write(io, "k")
        for i in 1:N_bands
            write(io, ",E_band_$i")
        end
        for (c, d) in zip(qn_sectors_gs, χ_gs)
            for j in 1:d
                write(io, ",e$j qn$(c.charge)")
            end
        end
        write(io, "\n")

        # data
        for i in 1:length(momenta)
            k = momenta[i]
            write(io, @sprintf("%.4f", k))
            for j in 1:N_bands
                write(io, @sprintf(",%.6f", real(E_excitations[i, j])))
            end

            es_k = get(entanglement_spectra, i, zeros(total_bd))
            for j in 1:total_bd
                write(io, @sprintf(",%.3f", j <= length(es_k) ? es_k[j] : 0.0))
            end

            write(io, "\n")
        end
    end

    println("Total runtime: ", time() - initial_time, " seconds")

end
