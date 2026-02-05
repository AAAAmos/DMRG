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
    Jnn, Jnnn, DMI, h = 1, 0.1, 0.0, 0.1

    obc_x = false 
    obc_y = false
    Ox, Oy = "f", "f"

#  -- Temperature --
    # beta_list = [0, 0.4, 0.5, 0.6, 0.7, 1, 1.3, 2, 3, 4, 5, 7, 10]
    beta_list = [0, 0.4, 0.5, 0.8, 1]
    d_beta = 0.01

#  -- numerical setup ---
    dmrg_linkdim = 10
    dmrg_maxdim = [200, 200, 200, 200, 200]
    maxdim = 256
    QN_conservation = true

#  -- evolution accuracy --
    psi_cutoff = 1E-10
    println("State cutoff: ", psi_cutoff)
    
    E_file = @sprintf(
        "./HC_data/%.i%.i_DM%.2f_Ox%s_Oy%s_psi%.i_db%.2f_QNt_k%.i.csv",
        N, M, DMI, Ox, Oy, Int(log10(psi_cutoff)), d_beta, maxdim
    )

# --- T=0 ---

#  -- physical states --

    sites = siteinds("S=1/2", N*M*2; conserve_sz = QN_conservation)
    states = ["Up" for n in 1:N*M*2]
    psi = MPS(Float64, sites, states)

    H, r = H_HC(N, M, Jnn, Jnnn, DMI, h; auc=false, obc_x, obc_y)
    H = MPO(H, sites)

#  -- dmrg --
    E0, psi0 = dmrg(H, psi; 
        nsweeps=10, maxdim=dmrg_maxdim, cutoff=psi_cutoff, outputlevel=1
    )
    mean_sz = mean(expect(psi0, "Sz"))

    open(E_file, "w") do io 
        write(io, "beta,E,Sz,dim\n")
        d = @sprintf("%.i,%.10f,%.5f,%.i\n", 100000, E0, mean_sz, maxlinkdim(psi0))
        write(io, d)
    end

    Psi_file = @sprintf(
        "./HC_data/%.i%.i_DM%.2f_Ox%s_Oy%s_GS_k%.i_psi.jld2",
        N, M, DMI, Ox, Oy, maxdim
    )
    save_object(Psi_file, psi0)
    Sites_file = @sprintf(
        "./HC_data/%.i%.i_DM%.2f_Ox%s_Oy%s_GS_k%.i_sites.jld2",
        N, M, DMI, Ox, Oy, maxdim
    )
    save_object(Sites_file, sites)

# --- finite T ---
    
#  -- aucillary states --
    psi_beta, sites = trivial_state(N*M*2*2, QN=QN_conservation) # |psi(beta=0)>
    H, r = H_HC(N, M, Jnn, Jnnn, DMI, h; auc=true)
    H = MPO(H, sites)

    for i in 1:length(beta_list)-1

        b = beta_list[i+1]-beta_list[i]
        beta = beta_list[i+1]
        nsweeps = round(Int, b/d_beta)

        if maxlinkdim(psi_beta) == maxdim
            nsites = 1
        else
            nsites = 2
        end

        psi_beta = tdvp(
            H, -b/2, psi_beta; 
            nsweeps=nsweeps, maxdim=maxdim, normalize=true, nsite=nsites, cutoff=psi_cutoff
            , outputlevel=1
        )
        println("Process peak RSS (MB): ", Sys.maxrss()/1.04E6)
        println("Cooldown fin.")
        println("Beta = ", beta)

#   - Save psi -
        Psi_file = @sprintf(
            "./HC_data/%.i%.i_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_psi.jld2",
            N, M, DMI, Ox, Oy, beta, maxdim
        )
        save_object(Psi_file, psi_beta)
        Sites_file = @sprintf(
            "./HC_data/%.i%.i_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_sites.jld2",
            N, M, DMI, Ox, Oy, beta, maxdim
        )
        save_object(Sites_file, sites)

#   - Mz, Purity, E -
        Mz = expect(psi_beta, "Sz")[2]
        dim = maxlinkdim(psi_beta)

        psi_h = apply(H, psi_beta; cutoff=psi_cutoff)
        E_beta = complex(0.0, 0.0)
        E_beta += inner(psi_beta, psi_h)

        open(E_file, "a") do io 
            d = @sprintf("%.4f,%.10f,%.5f,%.i\n", beta, E_beta.re, Mz, dim)
            write(io, d)
        end
    end

    println("Total computational time: $(time()-Total_time)")
end