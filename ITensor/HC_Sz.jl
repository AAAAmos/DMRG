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
    N = 12
    M = 6
    # Jnn, Jnnn, DMI, h = 1, -0.01, 0.2, 0.1
    # ani = 0.2
    Jnn, Jnnn, DMI, h = 1, 0.1, 0.1, 0.1
    ani = 0.

    # Spin !!
    # filename trivial_state input

    obc_x = true 
    obc_y = false 
    Ox, Oy = "t", "f"

#  -- Spin current --
    SC = false

    if SC 

        SCsites = []

        for col in 0:N-1
            x = M + col*2M
            append!(SCsites, [[x, x+2], [x+1, x-1]])
        end
        
        SCsites = unique(SCsites)
        @show SCsites

        # finite T:
        SCsites = SCsites.*2
    end

#  -- Temperature --
    # beta_list = [0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1, 1.1,
    #  1.2, 1.4, 1.7, 2, 2.2, 2.4, 2.7, 3, 3.4, 4, 5, 7]
    # h=0.1
    beta_list = [0, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
     2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3,
     3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 4,
     4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 5, 5.4, 6, 7]
    # h = 0.5
    # beta_list = [0, 0.1, 0.15, 0.17, 0.2, 0.23, 0.3, 0.33, 
    #  0.4, 0.43, 0.45, 0.47, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 
    #  1, 1.1, 1.3, 1.5, 1.7, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 
    #  3, 3.3, 3.7, 4, 4.3, 4.7, 5]
    # h=1.0
    # beta_list = [0, 0.1, 0.15, 0.17, 0.2, 0.23, 0.25, 0.3, 0.33, 0.35, 0.37, 
    #  0.4, 0.43, 0.45, 0.47, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 
    #  1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2, 2.1, 2.2, 2.3, 2.5, 2.7, 3, 3.3, 4, 5]
    # h=2.0
    # beta_list = [0, 0.1, 0.15, 0.17,
    #  0.2, 0.21, 0.22, 0.23, 0.25, 0.27, 0.3, 0.31, 0.32, 0.33, 0.35, 0.37, 
    #  0.4, 0.42, 0.46, 0.5, 0.53, 0.6, 0.63, 0.7, 0.73, 0.8, 0.9, 1, 1.2, 1.5, 1.7, 2]
    # Spin 1
    # beta_list = [0, 0.1, 0.15, 0.17,
    #  0.2, 0.22, 0.25, 0.27, 0.3, 0.31, 0.32, 0.33, 0.35, 0.37, 0.4, 0.41, 0.42, 0.45, 0.47, 
    #  0.5, 0.51, 0.53, 0.57, 0.6, 0.63, 0.67, 0.7, 0.73, 0.77, 0.8, 0.84, 0.9, 0.95, 1, 1.1, 1.2, 1.3, 1.5, 1.7, 
    #  2, 2.3, 2.5, 3]
    # Spin 3/2
    # beta_list = [0, 0.1, 0.15, 0.17,
    #  0.2, 0.22, 0.25, 0.27, 0.3, 0.31, 0.32, 0.33, 0.35, 0.37, 
    #  0.4, 0.41, 0.42, 0.45, 0.47, 0.5, 0.51, 0.53, 0.57, 0.6, 0.63, 0.67, 0.7, 0.73, 0.8, 0.9, 1, 1.2, 1.5, 1.7, 
    #  2, 2.3, 2.5, 3]
    # beta_list = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1, 1.1, 1.3, 1.6, 1.8]

    # save_list = []
    # save_list = [5, 3, 2, 1.7, 1.5, 1.3, 1.1, 1]
    # save_list = [0.33, 0.5, 0.7, 1, 1.1, 1.3, 1.5, 1.7, 2, 3]
    save_list = [1, 1.5, 2, 2.5, 2.8, 3, 3.3, 4, 5, 7]
    # save_list = [1.6, 2, 2.3, 2.5, 3, 
    #  3.1, 3.2, 3.3, 3.5, 3.7, 4, 
    #  4.1, 4.2, 4.4, 4.6, 4.8, 5, 7]
    # save_list = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3,
    #  3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 4,
    #  4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 5]
    d_beta = 0.01

#  -- numerical setup ---
    # dmrg_linkdim = 10
    # dmrg_maxdim = [200, 200, 200, 200, 200]
    maxdim = 15
    QN_conservation = true

#  -- evolution accuracy --
    psi_cutoff = 1E-10
    println("State cutoff: ", psi_cutoff)
    

# --- Data file ---

    E_file = @sprintf(
        "./HC_data/%.i%.i_S0.5_nnn%.2f_DM%.2f_h%.2f_Ox%s_Oy%s_psi%.i_db%.2f_QNt_k%.i.csv",
        N, M, Jnnn, DMI, h, Ox, Oy, Int(log10(psi_cutoff)), d_beta, maxdim
    )
    println("Output file: ", E_file)
    
    open(E_file, "w") do io 
        if obc_x 
            # Sz profile
            write(io, "beta,E,Sz,M2,dim")
            for i in 1:2M:2N*M
                write(io, ",Sz$i")
            end

            if SC
                for pairs in SCsites 
                    d = @sprintf(",S+%.i S-%.i", pairs[1], pairs[2])
                    write(io, d)
                end
            end

            write(io, "\n")
        else
            # Mean Sz
            write(io, "beta,E,Sz,M2,dim\n")
            # d = @sprintf("%.i,%.10f,%.5f,%.i\n", 100000, E0, mean_sz, maxlinkdim(psi0))
            # write(io, d)
        end
    end


# --- finite T ---
    
#  -- aucillary states --
    psi_beta, sites = trivial_state(SpinHalf(), N*M*2*2, QN=QN_conservation) # |psi(beta=0)>
    H, _, _ = H_HC(N, M, Jnn, Jnnn, DMI, h, ani; anc=true, obc_x, obc_y)
    H = MPO(H, sites)

    M1OP = MPO(M1_op(N*M*2*2), sites)
    M2OP = MPO(M2_op(N*M*2*2), sites)
    
    if SC
        Op_list = []
        for pairs in SCsites 
            os = OpSum()
            os += 1.0, "S+", pairs[1], "S-", pairs[2]
            push!(Op_list, MPO(os, sites))
        end
    end

    for i in 1:length(beta_list)-1

        b = beta_list[i+1]-beta_list[i]
        beta = beta_list[i+1]
        nsweeps = round(Int, b/d_beta)

        nsites = (beta >= 0.5) && (maxlinkdim(psi_beta) == maxdim) ? 1 : 2

        psi_beta = tdvp(
            H, -b/2, psi_beta; 
            nsweeps=nsweeps, maxdim=maxdim, normalize=true, nsite=nsites, cutoff=psi_cutoff
            , outputlevel=1
        )

        println("Beta = ", beta)
        println("Process peak RSS (GB): ", Sys.maxrss()/1.07E9)
        println("Cooldown fin.")
        GC.gc()

#   - Save psi -
        if beta in save_list
            Psi_file = @sprintf(
                "./HC_data/Psi/%.i%.i_nnn%.2f_DM%.2f_h%.2f_ani%.2f_Ox%s_Oy%s_b%.2f_k%.i_cut%.i_psi.jld2",
                N, M, Jnnn, DMI, h, ani, Ox, Oy, beta, maxdim, Int(log10(psi_cutoff))
            )
            save_object(Psi_file, psi_beta)
            Sites_file = @sprintf(
                "./HC_data/Psi/%.i%.i_nnn%.2f_DM%.2f_h%.2f_ani%.2f_Ox%s_Oy%s_b%.2f_k%.i_cut%.i_sites.jld2",
                N, M, Jnnn, DMI, h, ani, Ox, Oy, beta, maxdim, Int(log10(psi_cutoff))
            )
            save_object(Sites_file, sites)
        end

#   - Mz, Purity, E -
        Mz = inner(psi_beta', M1OP, psi_beta; cutoff=psi_cutoff)
        M2 = inner(psi_beta', M2OP, psi_beta; cutoff=psi_cutoff)
        E_beta = inner(psi_beta', H, psi_beta; cutoff=psi_cutoff)
        
        dim = maxlinkdim(psi_beta)

        open(E_file, "a") do io
            if obc_x 
                # Sz profile
                Sz_i = expect(psi_beta, "Sz"; sites=2:4M:4N*M)
                
                d = @sprintf("%.4f,%.10f,%.10f,%.10f,%.i", beta, real(E_beta), real(Mz), real(M2), dim)
                write(io, d)
            
                for sz in Sz_i 
                    d = @sprintf(",%.6f", real(sz))
                    write(io, d)
                end

                if SC 
                    for op in Op_list
                        exp_val = inner(psi_beta', op, psi_beta; cutoff=psi_cutoff)
                        write(io, @sprintf(",%.8f", real(exp_val)))
                    end
                end

                write(io, "\n")
                
            else
                d = @sprintf("%.4f,%.10f,%.10f,%.10f,%.i\n", beta, real(E_beta), real(Mz), real(M2), dim)
                write(io, d)
            end
        end
        GC.gc()
    end

    println("Total computational time: $(time()-Total_time)")
end