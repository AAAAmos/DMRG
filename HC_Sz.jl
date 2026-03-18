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
    N = 2
    M = 2
    Jnn, Jnnn, DMI, h = 1, 0.1, 0., 0.1
    ani = 0.

    obc_x = false 
    obc_y = false
    Ox, Oy = "f", "f"

    # Spin Check: Filename trivial_state

#  -- Temperature --
    # beta_list = [0, 0.4, 0.5, 0.6, 0.7, 1, 1.3, 2, 3, 4, 5, 7, 10]
    beta_list = [0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1, 1.1,
     1.2, 1.4, 1.7, 2, 2.2, 2.4, 2.7, 3, 3.4, 4, 5, 7]
    # beta_list = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1, 1.1, 1.3, 1.6, 2, 2.5, 3, 4, 5, 7]
    # save_list = [0.4, 0.5, 0.8, 1]
    save_list = []
    d_beta = 0.01

#  -- numerical setup ---
    dmrg_sw = 5
    dmrg_linkdim = 256
    # dmrg_maxdim = [200, 200, 200, 200, 200]
    dmrg_maxdim = ones(Int, dmrg_sw) * dmrg_linkdim
    maxdim = 8
    QN_conservation = true

#  -- evolution accuracy --
    psi_cutoff = 1E-10
    println("State cutoff: ", psi_cutoff)
    
#  -- files --
    E_file = @sprintf(
        "./HC_data/%.i%.i_S0.5_nnn%.2f_DM%.2f_Ox%s_Oy%s_psi%.i_db%.2f_QNt_k%.i.csv",
        N, M, Jnnn, DMI, Ox, Oy, Int(log10(psi_cutoff)), d_beta, maxdim
    )

    write_when_maxdim_exceeds = 20
    write_path = "/home/amos1/tensornetwork/"

# --- T=0 ---

#  -- physical states --

    sites = siteinds("S=1/2", N*M*2; conserve_sz = QN_conservation)
    states = ["Up" for n in 1:N*M*2]
    psi = MPS(Float64, sites, states)

    H, r = H_HC(N, M, Jnn, Jnnn, DMI, h, ani; auc=false, obc_x, obc_y)
    H = MPO(H, sites)

#  -- dmrg --
    E0, psi0 = dmrg(H, psi; 
        nsweeps=10, maxdim=dmrg_maxdim, cutoff=psi_cutoff, outputlevel=1
    )
    # mean_sz = mean(expect(psi0, "Sz"))

    # open(E_file, "w") do io 
    #     if obc_y 
    #         # Sz profile
    #         write(io, "beta,E,Sz,M2,dim")
    #         for i in 1:M 
    #             write(io, ",Sz$i")
    #         end
    #         write(io, "\n")
    #     else
    #         # Mean Sz
    #         write(io, "beta,E,Sz,M2,dim\n")
    #         # d = @sprintf("%.i,%.10f,%.5f,%.i\n", 100000, E0, mean_sz, maxlinkdim(psi0))
    #         # write(io, d)
    #     end
    # end

    Psi_file = @sprintf(
        "./HC_data/Psi/%.i%.i_DM%.2f_Ox%s_Oy%s_GS_k%.i_QNt_psi.jld2",
        N, M, DMI, Ox, Oy, dmrg_linkdim
    )
    save_object(Psi_file, psi0)
    Sites_file = @sprintf(
        "./HC_data/Psi/%.i%.i_DM%.2f_Ox%s_Oy%s_GS_k%.i_QNt_sites.jld2",
        N, M, DMI, Ox, Oy, dmrg_linkdim
    )
    save_object(Sites_file, sites)

    # psi0 = nothing 

# --- finite T ---
    
#  -- aucillary states --
    # psi_beta, sites = trivial_state(SpinHalf(), N*M*2*2, QN=QN_conservation) # |psi(beta=0)>
    # H, r = H_HC(N, M, Jnn, Jnnn, DMI, h, ani; auc=true, obc_x, obc_y)
    # H = MPO(H, sites)
    # M1OP = MPO(M1_op(N*M*2), sites)
    # M2OP = MPO(M2_op(N*M*2), sites)

    # for i in 1:length(beta_list)-1

    #     b = beta_list[i+1]-beta_list[i]
    #     beta = beta_list[i+1]
    #     nsweeps = round(Int, b/d_beta)

    #     if maxlinkdim(psi_beta) == maxdim
    #         nsites = 1
    #     else
    #         nsites = 2
    #     end

    #     # psi_beta = expand(psi_beta, H; 
    #     #         alg="global_krylov", krylovdim=2, cutoff=psi_cutoff
    #     #     )
    #     # psi_beta = tdvp(
    #     #     H, -b/2, psi_beta; 
    #     #     nsweeps=nsweeps, maxdim=maxdim, normalize=true, nsite=1, cutoff=psi_cutoff
    #     #     , outputlevel=1
    #     # )
    #     psi_beta = tdvp(
    #         H, -b/2, psi_beta; 
    #         nsweeps=nsweeps, maxdim=maxdim, normalize=true, nsite=nsites, cutoff=psi_cutoff
    #         , outputlevel=1#, write_when_maxdim_exceeds #, write_path
    #     )
    #     println("Process peak RSS (MB): ", Sys.maxrss()/1.04E6)
    #     println("Cooldown fin.")
    #     println("Beta = ", beta)

#   - Save psi -
        # if beta in save_list
        #     Psi_file = @sprintf(
        #         "./HC_data/Psi/%.i%.i_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_psi.jld2",
        #         N, M, DMI, Ox, Oy, beta, maxdim
        #     )
        #     save_object(Psi_file, psi_beta)
        #     Sites_file = @sprintf(
        #         "./HC_data/Psi/%.i%.i_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_sites.jld2",
        #         N, M, DMI, Ox, Oy, beta, maxdim
        #     )
        #     save_object(Sites_file, sites)
        # end

#   - Mz, Purity, E -
    #     Mz = inner(psi_beta', M1OP, psi_beta)
    #     M2 = inner(psi_beta', M2OP, psi_beta)
    #     E_beta = inner(psi_beta', H, psi_beta; cutoff=psi_cutoff)

    #     dim = maxlinkdim(psi_beta)

    #     open(E_file, "a") do io
    #         if obc_y 
    #             # Sz profile
    #             Sz_i = expect(psi_beta, "Sz"; sites=1:M)
                
    #             d = @sprintf("%.4f,%.10f,%.10f,%.10f,%.i", beta, real(E_beta), real(Mz), real(M2), dim)
    #             write(io, d)
            
    #             for i in 1:M 
    #                 d = @sprintf(",%.6f", real(Sz_i[i]))
    #                 write(io, d)
    #             end
    #             write(io, "\n")
                
    #         else
    #             d = @sprintf("%.4f,%.10f,%.10f,%.10f,%.i\n", beta, real(E_beta), real(Mz), real(M2), dim)
    #             write(io, d)
    #         end
    #     end

    #     GC.gc()
    # end

    println("Total computational time: $(time()-Total_time)")
end