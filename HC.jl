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
    N = 2
    M = 2
    Jnn, Jnnn, DMI, h = 1, 0.1, 0.0, 0.1

    obc_x = false 
    obc_y = false
    Ox, Oy = "f", "f"

#  -- Temperature --
    beta = 1/1

#  -- momentum --
    #=  a_1, a_2 = [√(3), 0], [√(3), 3]/2
        K+: [2, 1]/3
        K-: [1, 2]/3
        M: [1, 1]/2
    =#
    k1, k2 = 1, 1
    Q = 2*pi*(
        k1/N * [1/√(3), -1/3] + 
        k2/M * [0, 2/3]
    )

#  -- numerical setup ---
    dmrg_linkdim = 10
    dmrg_maxdim = [200, 200, 200, 200, 200]
    maxdim = 256
    QN_conservation = true

#  -- evolution accuracy --
    psi_cutoff = 1E-10
    time_cutoff = 1E-10
    println("State cutoff: ", psi_cutoff)

#  -- Time step --
    dtau = 0.05
    tausweep = 1
    Tsteps = 100

# --- T=0 ---

#  -- physical states --

    # Psi_file = @sprintf(
    #     "./HC_data/%.i%.i_DM%.2f_Ox%s_Oy%s_GS_k%.i_psi.jld2",
    #     N, M, DMI, Ox, Oy, maxdim
    # )
    # Sites_file = @sprintf(
    #     "./HC_data/%.i%.i_DM%.2f_Ox%s_Oy%s_GS_k%.i_sites.jld2",
    #     N, M, DMI, Ox, Oy, maxdim
    # )
    # psi0 = load_object(Psi_file)
    # sites = load_object(Sites_file)

    # H, r = H_HC(N, M, Jnn, Jnnn, DMI, h; auc=false, obc_x, obc_y)
    # H = MPO(H, sites)

#  -- Real space, real time --
    # BLAS.set_num_threads(1)
    
    # S1, S2 = 1, 1

    # filename = @sprintf(
    #     "./HC_data/Chi_%.i%.i_DM%.2f_Ox%s_Oy%s_S%.iS%.i_GS_psi%.i_Time%.i_xx.csv",
    #     N, M, DMI, Ox, Oy, S1, S2,
    #     Int(log10(psi_cutoff)), dtau*Tsteps 
    # )

    # O1, O2 = "Sx", "Sx"
    # chi = chi_x_t(O1, O2, S1, S2, H, psi0, sites, Tsteps, dtau, filename; 
    #     cutoff=psi_cutoff, maxdim=maxdim, ns=tausweep
    # )

#  -- Momentum space, real time --
    # BLAS.set_num_threads(1)
    
    # filename = @sprintf(
    #     "./HC_data/Chi_%.i%.i_DM%.2f_Ox%s_Oy%s_Q%.i%.i_GS_psi%.i_Time%.i_xx.csv",
    #     N, M, DMI, Ox, Oy, k1, k2,
    #     Int(log10(psi_cutoff)), dtau*Tsteps 
    # )

    # O1, O2 = "Sx", "Sx"
    # chi = chi_t(O1, O2, Q, r, H, psi0, sites, Tsteps, dtau, filename; 
    #     cutoff=psi_cutoff, maxdim=maxdim, ns=tausweep
    # )

# --- finite T ---

#  -- aucillary states --
    # psi_beta, sites = trivial_state(N*M*2*2, QN=QN_conservation) # |psi(beta=0)>
    # H, r = H_HC(N, M, Jnn, Jnnn, DMI, h; auc=true, obc_x, obc_y)
    # H = MPO(H, sites)
    # beta_list = [0, 1]
    # for i in 1:length(beta_list)-1

    #     b = beta_list[i+1]-beta_list[i]
    #     nsweeps = round(Int, b/0.01)

    #     psi_beta = tdvp(
    #         H, -b/2, psi_beta; 
    #         nsweeps=nsweeps, maxdim=maxdim, normalize=true, nsite=2, cutoff=psi_cutoff
    #         , outputlevel=1
    #     )
    #     println("Cooldown fin.")
    #     beta = beta_list[i+1]
    #     println("Beta = ", beta)
    # end

    Psi_file = @sprintf(
        "./HC_data/%.i%.i_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_psi.jld2",
        N, M, DMI, Ox, Oy, beta, maxdim
    )
    Sites_file = @sprintf(
        "./HC_data/%.i%.i_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_sites.jld2",
        N, M, DMI, Ox, Oy, beta, maxdim
    )
    println("Read data from: ", Psi_file)

    psi_beta = load_object(Psi_file)
    sites = load_object(Sites_file)

    H, r = H_HC(N, M, Jnn, Jnnn, DMI, h; auc=true, obc_x, obc_y)
    H = MPO(H, sites)
    
#  -- Real space, real time --
    # BLAS.set_num_threads(1)
    
    # S1, S2 = 2, 2

    # filename = @sprintf(
    #     "./HC_data/Chi_%.i%.i_DM%.2f_Ox%s_Oy%s_S%.iS%.i_beta%.2f_psi%.i_Time%.i_xx.csv",
    #     N, M, DMI, Ox, Oy, S1-1, S2-1, beta,
    #     Int(log10(psi_cutoff)), dtau*Tsteps 
    # )

    # O1, O2 = "Sx", "Sx"
    # chi = chi_x_t_FT(O1, O2, S1, S2, H, psi_beta, sites, Tsteps, dtau, filename; 
    #     cutoff=psi_cutoff, maxdim=maxdim, ns=tausweep
    # )
    
#  -- k space, real time --
    BLAS.set_num_threads(1)
    
    filename = @sprintf(
        "./HC_data/Chi_%.i%.i_DM%.2f_Ox%s_Oy%s_Q%.i%.i_beta%.2f_psi%.i_Time%.i_zz.csv",
        N, M, DMI, Ox, Oy, k1, k2, beta,
        Int(log10(psi_cutoff)), dtau*Tsteps,  
    )

    O1, O2 = "Sz", "Sz"
    chi = chi_t_FT(O1, O2, Q, r, H, psi_beta, sites, Tsteps, dtau, filename; 
        cutoff=psi_cutoff, maxdim=maxdim, ns=tausweep
    )

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