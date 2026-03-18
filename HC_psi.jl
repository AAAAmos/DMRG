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
    N = 1
    M = 7
    Jnn, Jnnn, DMI, h = 1, 0.1, 0.2, 0.1

    obc_x = true 
    obc_y = true
    Ox, Oy = "t", "t"

#  -- Temperature --
    beta_list = [0, 0.35, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
     2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3,
     3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 4,
     4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 5]
    # save_list = [1.6, 1.8, 2,
    #  2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3, 
    #  3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 4, 
    #  4.1, 4.2, 4.4, 4.6, 4.8, 5]
    save_list = []
    d_beta = 0.01

#  -- numerical setup ---
    maxdim = 10
    QN_conservation = true

#  -- evolution accuracy --
    psi_cutoff = 1E-8
    println("State cutoff: ", psi_cutoff)
    
    E_file = @sprintf(
        "./HC_data/%.i%.i_nnn%.2f_DM%.2f_Ox%s_Oy%s_psi%.i_db%.2f_QNt_k%.i.csv",
        N, M, Jnnn, DMI, Ox, Oy, Int(log10(psi_cutoff)), d_beta, maxdim
    )

    open(E_file, "w") do io 
        write(io, "beta,E,Sz,M2,dim\n")
    end

# --- finite T ---
    
#  -- aucillary states --
    psi_beta, sites = trivial_state(SpinOne(), N*M*2*2, QN=QN_conservation) 
    H, r = H_HC(N, M, Jnn, Jnnn, DMI, h; auc=true, obc_x, obc_y) 
    H = MPO(H, sites)

    for i in 1:length(beta_list)-1

        b = beta_list[i+1]-beta_list[i]
        beta = beta_list[i+1]
        nsweeps = round(Int, b/d_beta)

        if (beta >= 0.1) && (maxlinkdim(psi_beta) == maxdim)
            nsites = 1
        else
            nsites = 2
        end

        psi_beta = tdvp(
            H, -b/2, psi_beta; 
            nsweeps=nsweeps, maxdim=maxdim, normalize=true, nsite=nsites, cutoff=psi_cutoff
            , outputlevel=1
        )
        println("Process peak RSS (GB): ", Sys.maxrss()/1.07E9)
        println("Cooldown fin.")
        println("Beta = ", beta)

#   - Save psi -
        if beta in save_list
            Psi_file = @sprintf(
                "./HC_data/Psi/%.i%.i_nnn%.2f_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_psi.jld2",
                N, M, Jnnn, DMI, Ox, Oy, beta, maxdim
            )
            save_object(Psi_file, psi_beta)
            Sites_file = @sprintf(
                "./HC_data/Psi/%.i%.i_nnn%.2f_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_sites.jld2",
                N, M, Jnnn, DMI, Ox, Oy, beta, maxdim
            )
            save_object(Sites_file, sites)
        end

#   - Mz, Purity, E -
        M1OP = MPO(M1_op(N*M*2), sites)
        Mz = complex(0.0, 0.0)
        Mz += inner(psi_beta', M1OP, psi_beta)

        M2OP = MPO(M2_op(N*M*2), sites)
        M2 = complex(0.0, 0.0)
        M2 += inner(psi_beta', M2OP, psi_beta)
        
        dim = maxlinkdim(psi_beta)

        psi_h = apply(H, psi_beta; cutoff=psi_cutoff)
        E_beta = complex(0.0, 0.0)
        E_beta += inner(psi_beta, psi_h)

        open(E_file, "a") do io 
            d = @sprintf("%.4f,%.10f,%.5f,%.5f,%.i\n", beta, E_beta.re, Mz.re, M2.re, dim)
            write(io, d)
        end
    end

    println("Total computational time: $(time()-Total_time)")
end