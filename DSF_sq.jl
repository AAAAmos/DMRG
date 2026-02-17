using ITensors
using ITensorMPS
using Printf: @sprintf
using JLD2

function res(x, M) # same as % in python
    return mod(x-1, M)+1
end

function H_sq(n, m, h; auc=false, obc_x=false, obc_y=false)

    col = m
    N = n*col 
    J = 1
    Jz = 1

    r = Any[] # real space position
    a_1, a_2 = [1, 0], [0, 1]

    os = OpSum()

    # aucillary condition
    X = (auc) ? 2 : 1

    for a in 0:n-1 
        for b in 1:m 

            # build basis
            if auc 
                push!(r, 0)
            end
            push!(r, a*a_1 + b*a_2) 

            # index
            i = col*a + b
            ii = i*X

            # Zeeman
            os .+= -h, "Sz", ii 

            # up
            O_ = (obc_y && b+1>col) ? 0 : 1
            os .+= -Jz*O_/2, "Sz", ii, "Sz", (col*a+res(b+1, col))*X
            os .+= -J*O_/4, "S+", ii, "S-", (col*a+res(b+1, col))*X
            os .+= -J*O_/4, "S-", ii, "S+", (col*a+res(b+1, col))*X

            # dw
            O_ = (obc_y && b-1<0) ? 0 : 1
            os .+= -Jz*O_/2, "Sz", ii, "Sz", (col*a+res(b-1, col))*X
            os .+= -J*O_/4, "S+", ii, "S-", (col*a+res(b-1, col))*X
            os .+= -J*O_/4, "S-", ii, "S+", (col*a+res(b-1, col))*X

            O_ = (obc_x && i-col<0) ? 0 : 1
            os .+= -Jz*O_/2, "Sz", ii, "Sz", res(i-col, N)*X
            os .+= -J*O_/4, "S+", ii, "S-", res(i-col, N)*X
            os .+= -J*O_/4, "S-", ii, "S+", res(i-col, N)*X

            O_ = (obc_x && i+col>N) ? 0 : 1
            os .+= -Jz*O_/2, "Sz", ii, "Sz", res(i+col, N)*X
            os .+= -J*O_/4, "S+", ii, "S-", res(i+col, N)*X
            os .+= -J*O_/4, "S-", ii, "S+", res(i+col, N)*X

        end
    end
    
    return os, r
end

function M1_op(N)

    os = OpSum()
    for i in 2:2:N
        os += 1.0/(N/2), "Sz", i
    end
    return os
end

function M2_op(N)

    os = OpSum()
    N2 = (N/2)^2
    for i in 2:2:N
        for j in 2:2:N
            os += 1.0/N2, "Sz", i, "Sz", j
        end
    end
    return os
end

function trivial_state(N; QN=false)

    sites = siteinds("S=1/2", N; conserve_qns=QN, qnname_sz="TotalSz")

    states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi = MPS(Float64, sites, states)
    # psi = MPS(Float64, sites, "Up")

    # construct trivial state at beta=0
    # map |Up, Down> to 1/√2(|Up, Down> - |Down, Up>)
    gates = ITensor[]
    for j in 1:2:N-1
        s1 = siteind(psi, j)
        s2 = siteind(psi, j+1)
        
        # Create an operator tensor with 4 indices: (s1, s2) and (s1', s2')
        g = ITensor(dag(s1), dag(s2), s1', s2')
        # Index 1 = Up, Index 2 = Down
        g[s1=>1, s2=>2, s1'=>1, s2'=>2] = 1.0 / sqrt(2)
        g[s1=>1, s2=>2, s1'=>2, s2'=>1] = 1.0 / sqrt(2)

        push!(gates, g)
    end

    psi = apply(gates, psi; cutoff=1e-10)
    psi = noprime(psi)

    println("EPR state fin.")
    return psi, sites
end


let 
    Total_time = time()

#  -- Physical parameter setup ---
    N = 2
    M = 3
    h = 0.1

    obc_x = false 
    obc_y = false
    Ox, Oy = "f", "f"

#  -- Temperature --
    beta_list = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1, 1.2, 1.6, 2, 2.5, 3, 4, 5, 7]
    # beta_list = [0, 0.4, 0.5, 0.8, 1]
    d_beta = 0.01

#  -- numerical setup ---
    dmrg_linkdim = 10
    dmrg_maxdim = [200, 200, 200, 200, 200]
    maxdim = 64

#  -- evolution accuracy --
    psi_cutoff = 1E-10
    println("State cutoff: ", psi_cutoff)
    
    E_file = @sprintf(
        # "./Sq_data/long/Ising_%.i%.i_h%.2f_Ox%s_Oy%s_psi%.i_db%.2f_QNt_mk%.i.csv",
        "./Sq_data/%.i%.i_h%.2f_Ox%s_Oy%s_psi%.i_db%.2f_QNt_k%.i.csv",
        N, M, h, Ox, Oy, Int(log10(psi_cutoff)), d_beta, maxdim
    )

# --- finite T ---
    
    open(E_file, "w") do io 
        write(io, "beta,E,Sz,M2,dim\n")
    end

#  -- aucillary states --
    psi_beta, sites = trivial_state(N*M*2, QN=true) # |psi(beta=0)>
    H, r = H_sq(N, M, h; auc=true)
    H = MPO(H, sites)

    for i in 1:length(beta_list)-1

        b = beta_list[i+1]-beta_list[i]
        beta = beta_list[i+1]
        nsweeps = round(Int, b/d_beta)

        if (maxlinkdim(psi_beta) == maxdim)# & (i!=1)
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
        # Psi_file = @sprintf(
        #     "./HC_data/%.i%.i_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_psi.jld2",
        #     N, M, DMI, Ox, Oy, beta, maxdim
        # )
        # save_object(Psi_file, psi_beta)
        # Sites_file = @sprintf(
        #     "./HC_data/%.i%.i_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_sites.jld2",
        #     N, M, DMI, Ox, Oy, beta, maxdim
        # )
        # save_object(Sites_file, sites)

#   - Mz, Purity, E -
        # Mz = expect(psi_beta, "Sz")[2]
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
            d = @sprintf("%.4f,%.10f,%.10f,%.10f,%.i\n", beta, E_beta.re, Mz, M2, dim)
            write(io, d)
        end
    end

    println("Total computational time: $(time()-Total_time)")
end