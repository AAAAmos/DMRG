using ITensors
using ITensorMPS
using Printf: @sprintf
using Statistics: mean
using LinearAlgebra: BLAS
using JLD2
include("DSF_hc.jl")

# H_HC(2, 2, 1, 0.1, 0.1, 0.1, false, true, true)

let 
    Total_time = time()

#  -- Physical parameter setup ---
    N = 6
    M = 6
    Jnn, Jnnn, DMI, h = 1, 0.1, 0.2, 0.1
    ani = 0.

    obc_x = false 
    obc_y = false
    Ox, Oy = "f", "f"

    O1, O2 = "S+", "S-"
    operators = "+-"

    col_x = 4
    centerA, centerB = 1+2*M*(col_x-1), 2+2*M*(col_x-1)

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
    # k1, k2 = 2, 1
    # Q = 2*pi*(
    #     k1/N * [1/√(3), -1/3] + 
    #     k2/M * [0, 2/3]
    # )
    # Q_text = "Q$k1$k2"

    Q = 2*pi*(
        2/3 * [1/√(3), -1/3] + 
        1/3 * [0, 2/3]
    )
    Q_text = "K+"

#  -- numerical setup ---
    dmrg_sw = 5
    dmrg_linkdim = 256
    dmrg_maxdim = ones(Int, dmrg_sw) * dmrg_linkdim
    maxdim = 10

#  -- evolution accuracy --
    psi_cutoff = 1E-10
    time_cutoff = 1E-10
    println("State cutoff: ", psi_cutoff)

#  -- Time step --
    dtau = 0.05
    tausweep = 1
    Tsteps = 600*4

# --- T=0 ---

#  -- physical states --

    sites = siteinds("S=1/2", N*M*2; conserve_sz = true)
    states = ["Up" for n in 1:N*M*2]
    psi = MPS(Float64, sites, states)

#  -- dmrg --
    H, r = H_HC(N, M, Jnn, Jnnn, DMI, h, ani; auc=false, obc_x, obc_y)
    H = MPO(H, sites)

    E0, psi0 = dmrg(H, psi; 
        nsweeps=dmrg_sw, maxdim=dmrg_maxdim, cutoff=psi_cutoff, outputlevel=1
    )

#  -- Real space, real time --
    # BLAS.set_num_threads(1)

    # S1, S2 = 1+28, 7+28

    # filename = @sprintf(
    #     "./HC_data/Chi_%.i%.i_nnn%.2f_DM%.2f_Ox%s_Oy%s_S%.iS%.i_GS_psi%.i_dT%.2f_k%.i_%s_2T_expand3.csv",
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

    filename = @sprintf(
        "./HC_data/Chi_%.i%.i_nnn%.2f_DM%.2f_Ox%s_Oy%s_AB%.i%.i_%s_GS_psi%.i_Time%.i_k%.i_%s.csv",
        N, M, Jnnn, DMI, Ox, Oy, centerA, centerB, Q_text, 
        Int(log10(psi_cutoff)), dtau*Tsteps, maxdim, operators
    )

    chi = chi_t_old(O1, O2, Q, r, H, E0, psi0, sites, Tsteps, dtau, filename; 
        cutoff=time_cutoff, maxdim=maxdim, ns=tausweep, phi_t=false, centerA, centerB
    )

    # open(filename, "w") do io 
    #     write(io, "t,RS,IS\n")
    #     for t in -Tsteps:Tsteps
    #         i = t + Tsteps+1
    #         d = @sprintf("%.2f,%.10f,%.10f\n", t*dtau, chi[i].re, chi[i].im)
    #         write(io, d)
    #     end
    # end

# --- finite T ---

#  -- aucillary states --

    # Psi_file = @sprintf(
    #     "./HC_data/%.i%.i_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_psi.jld2",
    #     N, M, DMI, Ox, Oy, beta, maxdim
    # )
    # Sites_file = @sprintf(
    #     "./HC_data/%.i%.i_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_sites.jld2",
    #     N, M, DMI, Ox, Oy, beta, maxdim
    # )
    # println("Read data from: ", Psi_file)

    # psi_beta = load_object(Psi_file)
    # sites = load_object(Sites_file)

    # H, r = H_HC(N, M, Jnn, Jnnn, DMI, h; auc=true, obc_x, obc_y)
    # H = MPO(H, sites)
    
#  -- Real space, real time --
    # BLAS.set_num_threads(1)
    
    # S1, S2 = 2, 2

    # filename = @sprintf(
    #     "./HC_data/Chi_%.i%.i_DM%.2f_Ox%s_Oy%s_S%.iS%.i_beta%.2f_psi%.i_Time%.i_xx.csv",
    #     N, M, DMI, Ox, Oy, S1-1, S2-1, beta,
    #     Int(log10(psi_cutoff)), dtau*Tsteps 
    # )

    # chi = chi_x_t_FT(O1, O2, S1, S2, H, psi_beta, sites, Tsteps, dtau, filename; 
    #     cutoff=time_cutoff, maxdim=maxdim, ns=tausweep
    # )
    
#  -- k space, real time --
    # BLAS.set_num_threads(1)
    
    # filename = @sprintf(
    #     "./HC_data/Chi_%.i%.i_DM%.2f_Ox%s_Oy%s_Q%.i%.i_beta%.2f_psi%.i_Time%.i_%s.csv",
    #     N, M, DMI, Ox, Oy, k1, k2, beta, 
    #     Int(log10(psi_cutoff)), dtau*Tsteps, operators
    # )
    # println(filename)

    # chi = chi_t_FT(O1, O2, Q, r, H, psi_beta, sites, Tsteps, dtau, filename; 
    #     cutoff=time_cutoff, maxdim=maxdim, ns=tausweep
    # )

#   - 
    # open(filename, "w") do io 
    #     write(io, "t,RS,IS\n")
    #     for t in -Tsteps:Tsteps
    #         i = t + Tsteps+1
    #         d = @sprintf("%.2f,%.10f,%.10f\n", t*dtau, chi[i].re, chi[i].im)
    #         write(io, d)
    #     end
    # end

    println("Total computational time: $(time()-Total_time)")
end