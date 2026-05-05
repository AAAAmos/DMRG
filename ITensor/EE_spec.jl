using ITensors
using ITensorMPS
using Printf: @sprintf
using Statistics: mean
using LinearAlgebra: BLAS
using JLD2
include("DSF_hc.jl")

# H_HC(2, 2, 1, 0.1, 0.1, 0.1, false, true, true)


function Spec(auxiliary, psi, b, t, fname)
    return Spec(Val(auxiliary), psi, b, t, fname)
end

function Spec(auxiliary::Val{false}, psi, b, t, fname)
    # Move center to site b
    orthogonalize!(psi, b)

    wf_center, other = psi[b+1], psi[b]

    U, S, V = svd(wf_center, uniqueinds(wf_center,other))

    # spectrum = []
    
    open(fname, "a") do io 

        d = @sprintf("%.2f", t)
        write(io, d)
        for n in 1:dim(S, 1)
            lambda = S[n, n]
            # push!(spectrum, -2 * log(lambda))
            d = @sprintf(",%.10f", -2 * log(lambda))
            write(io, d)
        end
        write(io, "\n")
    end
    
end

function Spec(auxiliary::Val{true}, psi, b, t, fname)

    N_total = length(psi)
    N_phys = N_total ÷ 2
    
    mpo_tensors = Vector{ITensor}(undef, N_phys)

    for i in 1:N_phys

        a_idx = 2*i - 1
        p_idx = 2*i
        
        # We prime the physical site index to create the 'upper' leg of the MPO
        A = psi[a_idx]
        P = psi[p_idx]
        A_dag = dag(prime(A, "Link")) 
        P_dag = dag(prime(P, "Link"))
        
        # Prime the physical site, auxiliary cites will contract
        P_dag = prime(P_dag, siteinds(psi, p_idx))
        
        combined = A * P * A_dag * P_dag
        
        mpo_tensors[i] = combined
    end
    
    rho = MPO(mpo_tensors)

    orthogonalize!(rho, b)
    U, S, V = svd(rho[b], uniqueinds(rho[b], rho[b+1]))

    open(fname, "a") do io 

        d = @sprintf("%.2f", t)
        write(io, d)
        for n in 1:dim(S, 1)
            lambda = S[n, n]
            # push!(spectrum, -2 * log(lambda))
            d = @sprintf(",%.10f", -log(lambda))
            write(io, d)
        end
        write(io, "\n")
    end
    
end



function chi_t(O1, O2, H, psi0, sites, Tsteps, dt; cutoff=1e-10, maxdim=20, ns=1, centerA=1)

    N = length(psi0)
    println("Core A: ", centerA)
    println(" ")

    psi_t = copy(psi0)
    psi0[centerA] = noprime(op(O2, sites[centerA]) * psi0[centerA])

    # prepare for time evolution
    psi_SA_t = copy(psi0)

    psi_SA_t = expand(psi_SA_t, H; 
        alg="global_krylov", krylovdim=3, cutoff=cutoff
    )
    psi_t = expand(psi_t, H; 
        alg="global_krylov", krylovdim=3, cutoff=cutoff
    )

    Spec(psi_SA_t, 1, 0, "./test_Sm.txt")
    Spec(psi_t, 1, 0, "./test_psit.txt")

    nsitesA, nsitest = 1, 1

    for t in 1:Tsteps

        t_start = time()

        ol = (mod(t, 20) == 0) || (t<10) ? 1 : 0
        t1 = Threads.@spawn tdvp(H, -im*dt, psi_SA_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, cutoff=cutoff,
            nsite=nsitesA, outputlevel=ol
        )
        t2 = Threads.@spawn tdvp(H, -im*dt, psi_t; 
            nsweeps=ns, maxdim=maxdim, normalize=true, cutoff=cutoff,
            nsite=nsitest, outputlevel=ol
        )
        psi_SA_t, psi_t = (fetch(t1), fetch(t2))
        
        Spec(psi_SA_t, 1, t, "./test_Sm.txt")
        Spec(psi_t, 1, t, "./test_psit.txt")

        # sum over sites
        for j = 1:N-9

            psi_SA_t_Si = copy(psi_SA_t)
            psi_SA_t_Si[j] = noprime(op(O1, sites[j]) * psi_SA_t_Si[j])
            Spec(psi_SA_t_Si, 1, t, "./test_Spm.txt")

        end
        # chi_p[t, :] += chi_t_p 

        println("T step: $(t), Chi($(t*dt)) finished.")
        println("Loop Time spent: $(time()-t_start)")
        println("Process peak RSS (MB): ", Sys.maxrss()/1.04E6)
        
    end

    return 1
end

mutable struct EntanglementObserver <: AbstractObserver
end

function ITensorMPS.measure!(o::EntanglementObserver; bond, psi, half_sweep, kwargs...)
    println(bond)
    wf_center, other = half_sweep==1 ? (psi[bond+1],psi[bond]) : (psi[bond],psi[bond+1])
    U,S,V = svd(wf_center, uniqueinds(wf_center,other))
    SvN = 0.0
    for n=1:dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end
    println("  Entanglement across bond $bond = $SvN")
end


let 
    Total_time = time()

#  -- Physical parameter setup ---
    N = 2
    M = 2
    Jnn, Jnnn, DMI, h = 1, 0.1, 0.0, 0.1
    ani = 0.

    obc_x = false 
    obc_y = true 
    Ox, Oy = "f", "t"
    
#   - obc x, pbc y -
    # col_x = 4
    # centerA, centerB = 1+2*M*(col_x-1), 2+2*M*(col_x-1)
#   - obc y, pbc x -
    row_y = 1
    centerA, centerB = 1+2*(row_y-1), 2+2*(row_y-1)
    # -finite T-
    centerA *= 2
    centerB *= 2
#   - finite T need x2
    # centerA, centerB = 2, 4

    O1, O2 = "S+", "S-"
    operators = "+-"

    dynamic = false

#  -- Temperature --
    beta = 3
    # !!! FT 3 threads
    # !!! GS 2 threads

#  -- numerical setup ---
    dmrg_sw = 5
    dmrg_linkdim = 256
    dmrg_maxdim = ones(Int, dmrg_sw) * dmrg_linkdim
    maxdim = 200
    psidim = 8

#  -- evolution accuracy --
    psi_cutoff = 1E-10
    time_cutoff = 1E-10
    println("State cutoff: ", psi_cutoff)

#  -- Time step --
    dtau = 0.5
    tausweep = 10
    Tsteps = 40

# --- T>0 ---

    Psi_file = @sprintf(
        "./HC_data/Psi/%.i%.i_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_psi.jld2",
        N, M, DMI, Ox, Oy, beta, psidim
    )
    Sites_file = @sprintf(
        "./HC_data/Psi/%.i%.i_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_sites.jld2",
        N, M, DMI, Ox, Oy, beta, psidim
    )
    println("Read data from: ", Psi_file)

    psi_beta = load_object(Psi_file)
    sites = load_object(Sites_file)

    # H, r = H_HC(N, M, Jnn, Jnnn, DMI, h, ani; auc=true, obc_x, obc_y)
    # H = MPO(H, sites)

    mpo = Spec(true, psi_beta, 4, 0, "test.txt")
    @show mpo[1]

# --- T=0 ---

#  -- physical states --

    # sites = siteinds("S=1/2", N*M*2; conserve_sz = true)
    # states = ["Up" for n in 1:N*M*2]
    # psi = MPS(Float64, sites, states)

#  -- dmrg --
    # H, r = H_HC(N, M, Jnn, Jnnn, DMI, h, ani; auc=false, obc_x, obc_y)
    # H = MPO(H, sites)

    # # observer = EntanglementObserver()

    # E0, psi0 = dmrg(H, psi; 
    #     nsweeps=dmrg_sw, maxdim=dmrg_maxdim, cutoff=psi_cutoff, outputlevel=1
    # )

#  -- Momentum space, real time --
    # BLAS.set_num_threads(1)

    # b = 1
    # Spec(psi, b, t, fname)
    
    # chi = chi_t(O1, O2, H, psi0, sites, Tsteps, dtau; 
    #     cutoff=time_cutoff, maxdim, ns=tausweep
    # )

    println("Total computational time: $(time()-Total_time)")
end