using ITensors
using ITensorMPS
using Printf: @sprintf
using Statistics: mean
using LinearAlgebra: BLAS
using JLD2


function H_HC(n, m, J, j, D, h, ani; anc=false, obc_x=false, obc_y=false)
    
    res = (x, M) -> mod(x-1, M)+1

    col = m*2
    N = n*col 

    r = Any[] # real space position
    a_1, a_2 = [√(3), 0], [√(3), 3]/2

    os = OpSum()

    # ancilla condition
    X = (anc) ? 2 : 1

    for a in 0:n-1 
        for b in 1:m 

            # build basis
            if anc 
                push!(r, 0)
            end
            push!(r, a*a_1 + b*a_2) # A site
            if anc 
                push!(r, 0)
            end
            push!(r, a*a_1 + b*a_2 + [√(3), 1]/2) # B site

            # A index
            y = 2*b - 1
            i = col*a + y
            ii = i*X

            # Zeeman
            os .+= -h, "Sz", ii 
            os .+= -h, "Sz", ii+1*X
            
            # anisotropic
            os .+= -ani, "Sz", ii, "Sz", ii 
            os .+= -ani, "Sz", ii+1*X, "Sz", ii+1*X

            # A site NN
            os .+= -J/2, "Sz", ii, "Sz", ii+1*X
            os .+= -J/4, "S+", ii, "S-", ii+1*X
            os .+= -J/4, "S-", ii, "S+", ii+1*X

            O_ = (obc_y && y==1) ? 0 : 1
            os .+= -J*O_/2, "Sz", ii, "Sz", (col*a+res(y-1, col))*X
            os .+= -J*O_/4, "S+", ii, "S-", (col*a+res(y-1, col))*X
            os .+= -J*O_/4, "S-", ii, "S+", (col*a+res(y-1, col))*X
            # println("a NN", i, ": ", col*a+res(y-1, col), " O: ", O_)

            O_ = (obc_x && i+1-col<1) ? 0 : 1
            os .+= -J*O_/2, "Sz", ii, "Sz", res(i+1-col, N)*X
            os .+= -J*O_/4, "S+", ii, "S-", res(i+1-col, N)*X
            os .+= -J*O_/4, "S-", ii, "S+", res(i+1-col, N)*X
            # println("a NN", i, ": ", res(i+1-col, N), " O: ", O_)
            # println("a NN", i, ": ", i+1, col*a+res(y-1, col), res(i+1-col, N))

            # NNN
            O_ = (obc_y && y+2>col) ? 0 : 1
            os .+= -j*O_/2, "Sz", ii, "Sz", (col*a+res(y+2, col))*X
            os .+= (-j + im*D)*O_/4, "S+", ii, "S-", (col*a+res(y+2, col))*X
            os .+= (-j - im*D)*O_/4, "S-", ii, "S+", (col*a+res(y+2, col))*X
            # println("a NNN", ii, ": ", col*a+res(y+2, col), " O: ", O_)

            O_ = (obc_y && y-2<1) ? 0 : 1
            os .+= -j*O_/2, "Sz", ii, "Sz", (col*a+res(y-2, col))*X
            os .+= (-j - im*D)*O_/4, "S+", ii, "S-", (col*a+res(y-2, col))*X
            os .+= (-j + im*D)*O_/4, "S-", ii, "S+", (col*a+res(y-2, col))*X
            # println("a NNN", ii, ": ", col*a+res(y-2, col), " O: ", O_)
            
            O_ = (obc_x && i-col<1) ? 0 : 1
            os .+= -j*O_/2, "Sz", ii, "Sz", res(i-col, N)*X
            os .+= (-j + im*D)*O_/4, "S+", ii, "S-", res(i-col, N)*X
            os .+= (-j - im*D)*O_/4, "S-", ii, "S+", res(i-col, N)*X
            # println("a NNN", ii, ": ",  res(i-col, N), " O: ", O_)
            
            O_ = (obc_x && i+col>N) ? 0 : 1
            os .+= -j*O_/2, "Sz", ii, "Sz", res(i+col, N)*X
            os .+= (-j - im*D)*O_/4, "S+", ii, "S-", res(i+col, N)*X
            os .+= (-j + im*D)*O_/4, "S-", ii, "S+", res(i+col, N)*X
            # println("a NNN", ii, ": ",  res(i+col, N), " O: ", O_)

            O_x = (obc_x && a==n-1) ? 0 : 1
            O_y = (obc_y && y-2<1) ? 0 : 1
            os .+= (-j/2)*O_x*O_y, "Sz", ii, "Sz", res(col*(a+1)+res(y-2, col), N)*X
            os .+= (-j + im*D)*O_x*O_y/4, "S+", ii, "S-", res(col*(a+1)+res(y-2, col), N)*X
            os .+= (-j - im*D)*O_x*O_y/4, "S-", ii, "S+", res(col*(a+1)+res(y-2, col), N)*X
            # println("a NNN", ii, ": ", res(col*(a+1)+res(y-2, col), N), " O: ", O_x, O_y)

            O_x = (obc_x && a==0) ? 0 : 1
            O_y = (obc_y && y+2>col) ? 0 : 1
            os .+= (-j/2)*O_x*O_y, "Sz", ii, "Sz", res(col*(a-1)+res(y+2, col), N)*X
            os .+= (-j - im*D)*O_x*O_y/4, "S+", ii, "S-", res(col*(a-1)+res(y+2, col), N)*X
            os .+= (-j + im*D)*O_x*O_y/4, "S-", ii, "S+", res(col*(a-1)+res(y+2, col), N)*X
            # println("a NNN", ii, ": ", res(col*(a-1)+res(y+2, col), N), " O: ", O_x, O_y)
            # println("a NNN", ii, ": ", col*a+res(y+2, col), res(i-col, N), res(col*(a+1)+res(y-2, col), N))

            # B index
            y = 2*b
            i = col*a + y
            ii = i*X

            # B site NN
            os .+= -J/2, "Sz", ii, "Sz", ii-1*X
            os .+= -J/4, "S+", ii, "S-", ii-1*X
            os .+= -J/4, "S-", ii, "S+", ii-1*X

            O_ = (obc_y && y+1>col) ? 0 : 1
            os .+= -J*O_/2, "Sz", ii, "Sz", (col*a+res(y+1, col))*X
            os .+= -J*O_/4, "S+", ii, "S-", (col*a+res(y+1, col))*X
            os .+= -J*O_/4, "S-", ii, "S+", (col*a+res(y+1, col))*X
            # println("b NN", i, ": ", col*a+res(y+1, col), " O: ", O_)

            O_ = (obc_x && i-1+col>N) ? 0 : 1
            os .+= -J*O_/2, "Sz", ii, "Sz", res(i-1+col, N)*X
            os .+= -J*O_/4, "S+", ii, "S-", res(i-1+col, N)*X
            os .+= -J*O_/4, "S-", ii, "S+", res(i-1+col, N)*X
            # println("b NN", i, ": ", res(i-1+col, N), " O: ", O_)
            # println("b NN", i, ": ", i-1, col*a+res(y+1, col), res(i-1+col, N))

            # NNN
            O_ = (obc_y && y-2<1) ? 0 : 1
            os .+= (-j/2)*O_, "Sz", ii, "Sz", (col*a+res(y-2, col))*X
            os .+= (-j + im*D)*O_/4, "S+", ii, "S-", (col*a+res(y-2, col))*X
            os .+= (-j - im*D)*O_/4, "S-", ii, "S+", (col*a+res(y-2, col))*X
            # println("b NNN", i, ": ", col*a+res(y-2, col), " O: ", O_)
            
            O_ = (obc_y && y+2>col) ? 0 : 1
            os .+= (-j/2)*O_, "Sz", ii, "Sz", (col*a+res(y+2, col))*X
            os .+= (-j - im*D)*O_/4, "S+", ii, "S-", (col*a+res(y+2, col))*X
            os .+= (-j + im*D)*O_/4, "S-", ii, "S+", (col*a+res(y+2, col))*X
            # println("b NNN", i, ": ", col*a+res(y+2, col), " O: ", O_)
            
            O_ = (obc_x && i+col>N) ? 0 : 1
            os .+= (-j/2)*O_, "Sz", ii, "Sz", res(i+col, N)*X
            os .+= (-j + im*D)*O_/4, "S+", ii, "S-", res(i+col, N)*X
            os .+= (-j - im*D)*O_/4, "S-", ii, "S+", res(i+col, N)*X
            # println("b NNN", i, ": ", res(i+col, N), " O: ", O_)
            
            O_ = (obc_x && i-col<1) ? 0 : 1
            os .+= (-j/2)*O_, "Sz", ii, "Sz", res(i-col, N)*X
            os .+= (-j - im*D)*O_/4, "S+", ii, "S-", res(i-col, N)*X
            os .+= (-j + im*D)*O_/4, "S-", ii, "S+", res(i-col, N)*X
            # println("b NNN", i, ": ", res(i-col, N), " O: ", O_)

            O_x = (obc_x && a==0) ? 0 : 1
            O_y = (obc_y && y+2>col) ? 0 : 1
            os .+= (-j/2)*O_x*O_y, "Sz", ii, "Sz", res(col*(a-1)+res(y+2, col), N)*X
            os .+= (-j + im*D)*O_x*O_y/4, "S+", ii, "S-", res(col*(a-1)+res(y+2, col), N)*X
            os .+= (-j - im*D)*O_x*O_y/4, "S-", ii, "S+", res(col*(a-1)+res(y+2, col), N)*X
            # println("a NNN", ii, ": ", res(col*(a-1)+res(y+2, col), N), " O: ", O_x, O_y)
            
            O_x = (obc_x && a==n-1) ? 0 : 1
            O_y = (obc_y && y-2<1) ? 0 : 1
            os .+= (-j/2)*O_x*O_y, "Sz", ii, "Sz", res(col*(a+1)+res(y-2, col), N)*X
            os .+= (-j - im*D)*O_x*O_y/4, "S+", ii, "S-", res(col*(a+1)+res(y-2, col), N)*X
            os .+= (-j + im*D)*O_x*O_y/4, "S-", ii, "S+", res(col*(a+1)+res(y-2, col), N)*X
            # println("a NNN", ii, ": ", res(col*(a+1)+res(y-2, col), N), " O: ", O_x, O_y)
            # println("b NNN", i, ": ", col*a+res(y-2, col), res(i+col, N), res(col*(a-1)+res(y+2, col), N))

        end
    end
    
    return os, r
end

function chi_t_scan(ox, oy, centerA, centerB, N, M, O1, O2, Q_list, r, H, E0, psi0, sites, Tsteps, dt, filenames; kwargs...)
    if centerA == centerB
        return chi_t_scan(Val(ox), Val(oy), centerA, N, M, O1, O2, Q_list, r, H, E0, psi0, sites, Tsteps, dt, filenames; kwargs...)
    else
        return chi_t_scan(Val(ox), Val(oy), centerA, centerB, N, M, O1, O2, Q_list, r, H, E0, psi0, sites, Tsteps, dt, filenames; kwargs...)
    end
end

function chi_t_scan(ox::Val{true}, oy::Val{false}, center, N, M, O1, O2, Q_list, r, H, E0, psi0, sites, Tsteps, dt, filenames; cutoff=1e-10, maxdim=20, mindim=1, ns=1)

    N *= 2 # subsites
    Nk = length(Q_list)

    println("OBC in x, PBC in y. Correlation slices in x direction!")
    println("Core: ", center)
    println(" ")
    psi_prime = copy(psi0)
    psi_prime[center] = noprime(op(O2, sites[center]) * psi0[center])

    chi_t0 = zeros(ComplexF64, N, Nk)
    function sum_chi_helper_t0(n, m0)
        for m = m0:2:2M
            j = m+(n-1)*M
            psi_Sj_0 = copy(psi_prime)
            psi_Sj_0[j] = noprime(op(O1, sites[j]) * psi_Sj_0[j])

            for q = 1:Nk
                phaseK = dot(Q_list[q], (r[center]-r[j]))
                chi_t0[n+m0-1, q] += 0.5/√(N*M) * exp(-im * phaseK) * inner(psi0, psi_Sj_0)
            end
        end
        return 1
    end

    # t = 0
    for n = 1:2:N
        t1 = Threads.@spawn sum_chi_helper_t0(n, 1)
        t2 = Threads.@spawn sum_chi_helper_t0(n, 2)
        _, _ = (fetch(t1), fetch(t2))
    end

    # write data
    for q = 1:Nk
        open(filenames[q], "w") do io 
            write(io, "t")
            for n = 1:N 
                write(io, ",RS$n,IS$n")
            end
            write(io, "\n")

            d = @sprintf("%.2f", 0)
            write(io, d)
            for n = 1:N 
                d = @sprintf(",%.10f,%.10f", chi_t0[n, q].re, chi_t0[n, q].im)
                write(io, d)
            end
            write(io, "\n")
        end
    end


    # prepare for time evolution
    psi_Sj_t = copy(psi_prime)
    psi_prime, psi_Sj_0 = nothing, nothing

    # psi_Sj_t = expand(psi_Sj_t, H; 
    #     alg="global_krylov", krylovdim=3, 
    #     cutoff=cutoff
    # )
    nsites = 2

    for t in 1:Tsteps

        t_start = time()

        # psi_Sj_t = expand(psi_Sj_t, H; 
        #     alg="global_krylov", krylovdim=3, 
        #     cutoff=cutoff, mindim=mindim
        # )

        # ol = (mod(t, 20) == 1) ? 1 : 0
        ol = 1
        psi_Sj_t = tdvp(H, -im*dt, psi_Sj_t; 
            nsweeps=ns, maxdim=maxdim, mindim=mindim, normalize=false, 
            cutoff=cutoff,
            # noise=1,
            nsite=nsites, outputlevel=ol
        )

        # bonddim = maxlinkdim(psi_Sj_t)
        # nsites = (bonddim==maxdim) ? 1 : 2

        sum_time = time()

        chi_t_p = zeros(ComplexF64, N, Nk)
        function sum_chi_helper(n, m0)
            for m = m0:2:2M 
                j = m+(n-1)*M
                C = 0.5/√(N*M) * exp(im * E0 * t*dt) # from U^\dagger (t)

                O1_dag = dag(swapprime(op(O1, sites[j]), 0, 1))
                psi_t_O1dag = copy(psi0)
                psi_t_O1dag[j] = noprime(O1_dag * psi_t_O1dag[j])
                pSp = C * inner(psi_t_O1dag, psi_Sj_t)

                for q = 1:Nk
                    phaseK = dot(Q_list[q], (r[center]-r[j]))
                    chi_t_p[n+m0-1, q] += exp(-im * phaseK) * pSp
                end
            end 
            return 1
        end

        # sum over sites
        for n = 1:2:N
            t1 = Threads.@spawn sum_chi_helper(n, 1)
            t2 = Threads.@spawn sum_chi_helper(n, 2)
            _, _ = (fetch(t1), fetch(t2))
        end

        for q = 1:Nk
            open(filenames[q], "a") do io 
                d = @sprintf("%.2f", t*dt)
                write(io, d)
                for n = 1:N
                    d = @sprintf(",%.10f,%.10f", chi_t_p[n, q].re, chi_t_p[n, q].im)
                    write(io, d)
                end
                write(io, "\n")
            end
        end

    end

    return 1
end

let 
    Total_time = time()

#  -- Physical parameter setup ---
    # N >= M to has smaller entanglement
    N = 2
    M = 2
    Jnn, Jnnn, DMI, h = 2, 0., 0., 0.1
    # D=0 3618 edge
    ani = 0.

    obc_x = false 
    obc_y = false   
    Ox, Oy = "t", "f"

    O1, O2 = "S+", "S-"
    operators = "+-"

#   x obc, y pbc
    col_x = 9
    centerA, centerB = M+2*M*(col_x-1), M +2*M*(col_x-1)

#  -- momentum --
    Q_list = []
    Q_text = []
    n_k = 0

#   - K+ -
    Q = 2*pi*(
        2/3 * [1/√(3), -1/3] + 
        1/3 * [0, 2/3]
    )
    push!(Q_list, Q)
    push!(Q_text, "0_1.0K+")


    println("Momenta: ", Q_text)

#  -- numerical setup ---
    dmrg_sw = 5
    dmrg_linkdim = 5
    dmrg_maxdim = ones(Int, dmrg_sw) * dmrg_linkdim
    maxdim = 10
    mindim = 4

#  -- evolution accuracy --
    psi_cutoff = 1E-10
    time_cutoff = 0.0
    println("State cutoff: ", psi_cutoff)

#  -- Time step --
    dtau = 0.05
    tausweep = 1
    Tsteps = 10

# --- T=0 ---

#  -- physical states --

    sites = siteinds("S=1/2", N*M*2; conserve_sz = true)
    states = ["Up" for n in 1:N*M*2]
    states[1] = "Dn"
    # states[N*M*2] = "Dn"
    # states = [isodd(n) ? "Up" : "Dn" for n in 1:N*M*2]
    psi = MPS(Float64, sites, states)

#  -- dmrg --
    H, r = H_HC(N, M, Jnn, Jnnn, DMI, h, ani; anc=false, obc_x, obc_y)
    H = MPO(H, sites)

    E0, psi0 = dmrg(H, psi; 
        nsweeps=dmrg_sw, maxdim=dmrg_maxdim, mindim=mindim,
        cutoff=0.0, 
        outputlevel=1
    )

    GC.gc()


#  -- Momentum space, real time --
    # BLAS.set_num_threads(1)

    # filenames = []
    # for q = 1:length(Q_text)
    #     filename = @sprintf(
    #         "../HC_data/Chi_%.i%.i_nnn%.2f_DM%.2f_h%.2f_Ox%s_Oy%s_AB%.i%.i_scany_%s_GS_psi%.i_Time%.i_k%.i_%s_gauged.csv",
    #         N, M, Jnnn, DMI, h, Ox, Oy, centerA, centerB, Q_text[q], 
    #         Int(log10(psi_cutoff)), dtau*Tsteps, maxdim, operators
    #     )
    #     println(filename)
    #     push!(filenames, filename)
    # end

    # chi = chi_t_scan(obc_x, obc_y, centerA, centerB, N, M, O1, O2, Q_list, r, H, E0, psi0, sites, Tsteps, dtau, filenames; 
    #     cutoff=time_cutoff, maxdim=maxdim, mindim=mindim, ns=tausweep
    # )

    println("Total computational time: $(time()-Total_time)")
end