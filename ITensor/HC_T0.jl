using ITensors
using ITensorMPS
using Printf: @sprintf
using Statistics: mean
using LinearAlgebra: BLAS
using JLD2
include("DSF_hc.jl")

let 
    Total_time = time()

#  -- Physical parameter setup ---
    # N >= M to has smaller entanglement
    N = 24
    M = 12
    Jnn, Jnnn, DMI, h = 1, 0.1, 0.1, 0.1
    # D=0 3618 edge
    ani = 0.

    obc_x = true 
    obc_y = false   
    Ox, Oy = "t", "f"

    O1, O2 = "S+", "S-"
    operators = "+-"
    # O1, O2 = "S-", "S+"
    # operators = "-+"
#   x obc, y pbc
    col_x = 9
    centerA, centerB = M+2*M*(col_x-1), M +2*M*(col_x-1)
    # centerA, centerB = 19, 19 # M
#   x pbc, y obc
    # row_y = 1
    # centerA, centerB = 1+2*(row_y-1), 2+2*(row_y-1)
    # centerA, centerB = 1, 1

#  -- Temperature --
    beta = 1/1
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

#   - K- -
    # x_fractions = range(0, stop=3, length=N+1)
    # for q =1:length(x_fractions)
    #     Q = x_fractions[q] * 2*pi*(
    #         1/3 * [1/√(3), -1/3] + 
    #         2/3 * [0, 2/3]
    #     )
    #     push!(Q_list, Q)
    #     n_k += 1
    #     Qs = @sprintf("%s_%.2fK-", n_k, x_fractions[q])
    #     push!(Q_text, Qs)
    # end
    # Q = 2*pi*(
    #     1/3 * [1/√(3), -1/3] + 
    #     2/3 * [0, 2/3]
    # )
    # push!(Q_list, Q)
    # push!(Q_text, "1_1.0K-")

    # Q = -2*pi*(
    #     1/3 * [1/√(3), -1/3] + 
    #     2/3 * [0, 2/3]
    # )
    # push!(Q_list, Q)
    # push!(Q_text, "2_-1.0K-")

#   - K+ -
    Q = 2*pi*(
        2/3 * [1/√(3), -1/3] + 
        1/3 * [0, 2/3]
    )
    push!(Q_list, Q)
    push!(Q_text, "0_1.0K+")

    # Q = -2*pi*(
    #     2/3 * [1/√(3), -1/3] + 
    #     1/3 * [0, 2/3]
    # )
    # push!(Q_list, Q)
    # push!(Q_text, "4_-1.0K+")

    println("Momenta: ", Q_text)

#  -- numerical setup ---
    dmrg_sw = 5
    dmrg_linkdim = 10
    dmrg_maxdim = ones(Int, dmrg_sw) * dmrg_linkdim
    maxdim = 10
    mindim = 4

#  -- evolution accuracy --
    psi_cutoff = 1E-10
    time_cutoff = 1E-10
    println("State cutoff: ", psi_cutoff)

#  -- Time step --
    dtau = 0.05
    tausweep = 1
    Tsteps = 10

# --- T=0 ---

#  -- physical states --

    sites = siteinds("S=1/2", N*M*2; conserve_sz = true)
    states = ["Up" for n in 1:N*M*2]
    psi = MPS(Float64, sites, states)

#  -- dmrg --
    H, r = H_HC(N, M, Jnn, Jnnn, DMI, h, ani; anc=false, obc_x, obc_y)
    H = MPO(H, sites)

    E0, psi0 = dmrg(H, psi; 
        nsweeps=dmrg_sw, maxdim=dmrg_maxdim, mindim=mindim,
        # cutoff=psi_cutoff, 
        outputlevel=1
    )

    psi0 = expand(psi0, H; 
        alg="global_krylov", krylovdim=3
    )
    println(maxlinkdim(psi0))
    GC.gc()

#  -- Real space, real time --
    # BLAS.set_num_threads(1)

    # # S1, S2 = 29, 28
    # S1, S2 = 1, 42

    # filename = @sprintf(
    #     "./HC_data/Chi_%.i%.i_nnn%.2f_DM%.2f_Ox%s_Oy%s_S%.iS%.i_GS_psi%.i_dT%.2f_k%.i_%s_2T_expand.csv",
    #     N, M, Jnnn, DMI, Ox, Oy, S1, S2,
    #     Int(log10(psi_cutoff)), dtau, maxdim, operators
    # )
    # println(filename)

    # chi = chi_x_t(O1, O2, S1, S2, H, E0, psi0, sites, Tsteps, dtau, filename; 
    #     cutoff=time_cutoff, maxdim=maxdim, ns=tausweep
    # )

    # open(filename, "w") do io 
    #     write(io, "t,RS,IS\n")
    #     for t in -Tsteps:Tsteps
    #         i = t + Tsteps+1
    #         d = @sprintf("%.2f,%.10f,%.10f\n", t*dtau, chi[i].re, chi[i].im)
    #         write(io, d)
    #     end
    # end

#  -- Momentum space, real time --
    BLAS.set_num_threads(1)

    filenames = []
    for q = 1:length(Q_text)
        filename = @sprintf(
            "../HC_data/Chi_%.i%.i_nnn%.2f_DM%.2f_h%.2f_Ox%s_Oy%s_AB%.i%.i_scany_%s_GS_psi%.i_Time%.i_k%.i_%s_gauged.csv",
            N, M, Jnnn, DMI, h, Ox, Oy, centerA, centerB, Q_text[q], 
            Int(log10(psi_cutoff)), dtau*Tsteps, maxdim, operators
        )
        println(filename)
        push!(filenames, filename)
    end

    chi = chi_t_scan(obc_x, obc_y, centerA, centerB, N, M, O1, O2, Q_list, r, H, E0, psi0, sites, Tsteps, dtau, filenames; 
        cutoff=time_cutoff, maxdim=maxdim, mindim=mindim, ns=tausweep
    )
    # chi = chi_t(O1, O2, Q_list, r, H, E0, psi0, sites, Tsteps, dtau, filenames; 
    #     cutoff=time_cutoff, maxdim=maxdim, ns=tausweep, phi_t=false, centerA, centerB
    # )

    println("Total computational time: $(time()-Total_time)")
end