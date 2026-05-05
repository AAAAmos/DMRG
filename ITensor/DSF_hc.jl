using ITensors
using ITensorMPS
using LinearAlgebra: svd, dot

struct SpinHalf end 
struct SpinOne end 

function res(x, M) # same as % in python
    return mod(x-1, M)+1
end

function H_HC(n, m, J, j, D, h, ani; anc=false, obc_x=false, obc_y=false)

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

function trivial_state(::SpinHalf, N; QN=false)

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

    println("Spin Half EPR state fin.")
    return psi, sites
end

function trivial_state(::SpinOne, N; QN=false)

    sites = siteinds("S=1", N; conserve_qns=QN, qnname_sz="TotalSz")

    states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    psi = MPS(Float64, sites, states)

    gates = ITensor[]
    for j in 1:2:N-1
        s1 = siteind(psi, j)
        s2 = siteind(psi, j+1)
        
        # Create an operator tensor with 4 indices: (s1, s2) and (s1', s2')
        g = ITensor(dag(s1), dag(s2), s1', s2')
        g[s1=>"Up", s2=>"Dn", s1'=>"Up", s2'=>"Dn"] = 1.0 / sqrt(3)
        g[s1=>"Up", s2=>"Dn", s1'=>"Dn", s2'=>"Up"] = 1.0 / sqrt(3)
        g[s1=>"Up", s2=>"Dn", s1'=>"Z0", s2'=>"Z0"] = 1.0 / sqrt(3)

        push!(gates, g)
    end

    psi = apply(gates, psi; cutoff=1e-10)
    psi = noprime(psi)

    println("Spin 1 EPR state fin.")
    return psi, sites
end

function chi_t_old(O1, O2, Q, r, H, E0, psi0, sites, Tsteps, dt, filename; cutoff=1e-10, maxdim=20, ns=1, centerA=1, centerB=2, phi_t=false)

    chi_p = zeros(ComplexF64, Tsteps) # store the chi(t)

    N = length(psi0)
    println("Core A: ", centerA, "Core B: ", centerB)
    println(" ")
    psi_Aprime, psi_Bprime, psi_t = copy(psi0), copy(psi0), copy(psi0)
    psi_Aprime[centerA] = noprime(op(O2, sites[centerA]) * psi0[centerA])
    psi_Bprime[centerB] = noprime(op(O2, sites[centerB]) * psi0[centerB])

    # t = 0
    chi_t0 = complex(0.0, 0.0)
    for j = 1:N
        psi_Sj_0 = copy(psi_Aprime)
        psi_Sj_0[j] = noprime(op(O1, sites[j]) * psi_Sj_0[j])
        phaseK = dot(Q, (r[centerA]-r[j]))
        chi_t0 += 0.5/√(N) * exp(-im * phaseK) * inner(psi0, psi_Sj_0)
        psi_Sj_0 = copy(psi_Bprime)
        psi_Sj_0[j] = noprime(op(O1, sites[j]) * psi_Sj_0[j])
        phaseK = dot(Q, (r[centerB]-r[j]))
        chi_t0 += 0.5/√(N) * exp(-im * phaseK) * inner(psi0, psi_Sj_0)
    end

    # prepare for time evolution
    psi_SA_t, psi_SB_t = copy(psi_Aprime), copy(psi_Bprime)
    psi_Aprime, psi_Bprime = nothing, nothing

    # write data
    if phi_t
        phiphi = expect(psi_t, [1 0; 0 0])
        # phiphi = expect(psi_t, "Sz")

        open(filename, "w") do io 
            write(io, "t,RS,IS")
            for n in 1:N
                write(io, ",Spm$n")
            end
            write(io, "\n")

            d = @sprintf("%.2f,%.10f,%.10f", 0, chi_t0.re, chi_t0.im)
            write(io, d)
            for n in 1:N
                d = @sprintf(",%.6f", phiphi[n])
                write(io, d)
            end
            write(io, "\n")
        end
    else
        open(filename, "w") do io 
            write(io, "t,RS,IS\n")
            d = @sprintf("%.2f,%.10f,%.10f\n", 0, chi_t0.re, chi_t0.im)
            write(io, d)
        end
    end

    psi_SA_t = expand(psi_SA_t, H; 
        alg="global_krylov", krylovdim=3, cutoff=cutoff
    )
    psi_SB_t = expand(psi_SB_t, H; 
        alg="global_krylov", krylovdim=3, cutoff=cutoff
    )
    # psi_t =  expand(psi_t, H; 
    #     alg="global_krylov", krylovdim=3, cutoff=cutoff
    # )
    nsitesA, nsitesB = 1, 1

    for t in 1:Tsteps

        t_start = time()

        ol = (mod(t, 20) == 0) || (t<10) ? 1 : 0
        t1 = Threads.@spawn tdvp(H, -im*dt, psi_SA_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, cutoff=cutoff,
            nsite=nsitesA, outputlevel=ol
        )
        t2 = Threads.@spawn tdvp(H, -im*dt, psi_SB_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, cutoff=cutoff,
            nsite=nsitesB, outputlevel=ol
        )
        # t3 = Threads.@spawn tdvp(H, -im*dt, psi_t; 
        #     nsweeps=ns, maxdim=maxdim, normalize=false, cutoff=cutoff,
        #     nsite=nsitesA, outputlevel=ol
        # )
        # psi_SA_t, psi_SB_t, psi_t = (fetch(t1), fetch(t2), fetch(t3))
        psi_SA_t, psi_SB_t = (fetch(t1), fetch(t2))

        bonddimA, bonddimB = maxlinkdim(psi_SA_t), maxlinkdim(psi_SB_t)
        nsitesA = (bonddimA==maxdim) ? 1 : 2
        nsitesB = (bonddimB==maxdim) ? 1 : 2

        chi_t_p = complex(0.0, 0.0)
        # sum over sites
        for j = 1:N
            C = 0.5/√(N) * exp(im * E0 * t*dt) # from U^\dagger (t)

            psi_SA_t_Si = copy(psi_SA_t)
            psi_SA_t_Si[j] = noprime(op(O1, sites[j]) * psi_SA_t_Si[j])
            # momentum phase
            phaseK = dot(Q, (r[centerA]-r[j]))
            chi_t_p += C * exp(-im * phaseK) * inner(psi0, psi_SA_t_Si)
            
            psi_SB_t_Si = copy(psi_SB_t)
            psi_SB_t_Si[j] = noprime(op(O1, sites[j]) * psi_SB_t_Si[j])
            # momentum phase
            phaseK = dot(Q, (r[centerB]-r[j]))
            chi_t_p += C * exp(-im * phaseK) * inner(psi0, psi_SB_t_Si)
            
        end
        chi_p[t] = chi_t_p 

        if mod(t, 20) == 0
            println("T step: $(t), Chi($(t*dt)) finished.")
            println("Loop Time spent: $(time()-t_start)")
            println("Process peak RSS (MB): ", Sys.maxrss()/1.04E6)
        end

        # write data incase unexpected termination happend
        if phi_t
            phiphi = expect(psi_t, [1 0; 0 0])
            # phiphi = expect(psi_t, "Sz")

            open(filename, "a") do io 
                d = @sprintf("%.2f,%.10f,%.10f", t*dt, chi_t_p.re, chi_t_p.im)
                write(io, d)

                for n in 1:N
                    d = @sprintf(",%.6f", phiphi[n])
                    write(io, d)
                end
                write(io, "\n")
            end
        else
            open(filename, "a") do io 
                d = @sprintf("%.2f,%.10f,%.10f\n", t*dt, chi_t_p.re, chi_t_p.im)
                write(io, d)
            end
        end
    end

    # chi_n = reverse(conj(chi_p))
    # chi_n = reverse(chi_n)
    # chi = append!(chi_n, chi_t0, chi_p) # time grid from -T to +T

    return chi_p
end

function chi_t(O1, O2, Q_list, r, H, E0, psi0, sites, Tsteps, dt, filenames; cutoff=1e-10, maxdim=20, ns=1, centerA=1, centerB=2, phi_t=false)

    Nk = length(Q_list)

    chi_t0 = zeros(ComplexF64, Nk)
    # chi_p = zeros(ComplexF64, Tsteps, Nk) # store the chi(t)

    N = length(psi0)
    println("Core A: ", centerA, "Core B: ", centerB)
    println(" ")
    psi_Aprime, psi_Bprime, psi_t = copy(psi0), copy(psi0), copy(psi0)
    psi_Aprime[centerA] = noprime(op(O2, sites[centerA]) * psi0[centerA])
    psi_Bprime[centerB] = noprime(op(O2, sites[centerB]) * psi0[centerB])

    # t = 0
    for j = 1:N
        psi_Sja_0 = copy(psi_Aprime)
        psi_Sja_0[j] = noprime(op(O1, sites[j]) * psi_Sja_0[j])
        psi_Sjb_0 = copy(psi_Bprime)
        psi_Sjb_0[j] = noprime(op(O1, sites[j]) * psi_Sjb_0[j])

        for q = 1:Nk
            phaseK = dot(Q_list[q], (r[centerA]-r[j]))
            chi_t0[q] += 0.5/√(N) * exp(-im * phaseK) * inner(psi0, psi_Sja_0)
            phaseK = dot(Q_list[q], (r[centerB]-r[j]))
            chi_t0[q] += 0.5/√(N) * exp(-im * phaseK) * inner(psi0, psi_Sjb_0)
        end
    end

    # prepare for time evolution
    psi_SA_t, psi_SB_t = copy(psi_Aprime), copy(psi_Bprime)
    psi_Aprime, psi_Bprime = nothing, nothing
    psi_Sja_0, psi_Sjb_0 = nothing, nothing

    # write data
    for q = 1:Nk
        open(filenames[q], "w") do io 
            write(io, "t,RS,IS\n")
            d = @sprintf("%.2f,%.10f,%.10f\n", 0, chi_t0[q].re, chi_t0[q].im)
            write(io, d)
        end
    end

    psi_SA_t = expand(psi_SA_t, H; 
        alg="global_krylov", krylovdim=3, cutoff=cutoff
    )
    psi_SB_t = expand(psi_SB_t, H; 
        alg="global_krylov", krylovdim=3, cutoff=cutoff
    )
    nsitesA, nsitesB = 1, 1

    for t in 1:Tsteps

        t_start = time()

        ol = (mod(t, 20) == 0) || (t<10) ? 1 : 0
        t1 = Threads.@spawn tdvp(H, -im*dt, psi_SA_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, cutoff=cutoff,
            nsite=nsitesA, outputlevel=ol
        )
        t2 = Threads.@spawn tdvp(H, -im*dt, psi_SB_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, cutoff=cutoff,
            nsite=nsitesB, outputlevel=ol
        )
        psi_SA_t, psi_SB_t = (fetch(t1), fetch(t2))

        bonddimA, bonddimB = maxlinkdim(psi_SA_t), maxlinkdim(psi_SB_t)
        nsitesA = (bonddimA==maxdim) ? 1 : 2
        nsitesB = (bonddimB==maxdim) ? 1 : 2

        chi_t_p = zeros(ComplexF64, Nk)
        # sum over sites
        for j = 1:N
            C = 0.5/√(N) * exp(im * E0 * t*dt) # from U^\dagger (t)

            psi_SA_t_Si = copy(psi_SA_t)
            psi_SA_t_Si[j] = noprime(op(O1, sites[j]) * psi_SA_t_Si[j])
            pAp = C * inner(psi0, psi_SA_t_Si)
            psi_SB_t_Si = copy(psi_SB_t)
            psi_SB_t_Si[j] = noprime(op(O1, sites[j]) * psi_SB_t_Si[j])
            pBp = C * inner(psi0, psi_SB_t_Si)

            for q = 1:Nk
                phaseK = dot(Q_list[q], (r[centerA]-r[j]))
                chi_t_p[q] += exp(-im * phaseK) * pAp
                phaseK = dot(Q_list[q], (r[centerB]-r[j]))
                chi_t_p[q] += exp(-im * phaseK) * pBp
            end
        end
        # chi_p[t, :] += chi_t_p 

        if mod(t, 20) == 0
            println("T step: $(t), Chi($(t*dt)) finished.")
            println("Loop Time spent: $(time()-t_start)")
            println("Process peak RSS (MB): ", Sys.maxrss()/1.04E6)
        end

        for q = 1:Nk
            open(filenames[q], "a") do io 
                d = @sprintf("%.2f,%.10f,%.10f\n", t*dt, chi_t_p[q].re, chi_t_p[q].im)
                write(io, d)
            end
        end
        
    end

    return 1
end

function chi_t_scan(ox, oy, N, M, O1, O2, Q_list, r, H, E0, psi0, sites, Tsteps, dt, filenames; kwargs...)
    return chi_t_scan(Val(ox), Val(oy), N, M, O1, O2, Q_list, r, H, E0, psi0, sites, Tsteps, dt, filenames; kwargs...)
end

function chi_t_scan(ox::Val{false}, oy::Val{true}, N, M, O1, O2, Q_list, r, H, E0, psi0, sites, Tsteps, dt, filenames; cutoff=1e-10, maxdim=20, ns=1, centerA=1, centerB=2, phi_t=false)

    M *= 2 # subsites
    Nk = length(Q_list)

    println("PBC in x, OBC in y. Correlation slices in y direction!")
    println("Core A: ", centerA, "Core B: ", centerB)
    println(" ")
    psi_Aprime, psi_Bprime = copy(psi0), copy(psi0)
    psi_Aprime[centerA] = noprime(op(O2, sites[centerA]) * psi0[centerA])
    psi_Bprime[centerB] = noprime(op(O2, sites[centerB]) * psi0[centerB])

    chi_t0 = zeros(ComplexF64, M, Nk)

    # t = 0
    for m = 1:M
        for n = 1:N
            j = m+(n-1)*M
            psi_Sja_0 = copy(psi_Aprime)
            psi_Sja_0[j] = noprime(op(O1, sites[j]) * psi_Sja_0[j])
            psi_Sjb_0 = copy(psi_Bprime)
            psi_Sjb_0[j] = noprime(op(O1, sites[j]) * psi_Sjb_0[j])

            for q = 1:Nk
                phaseK = dot(Q_list[q], (r[centerA]-r[j]))
                chi_t0[m, q] += 0.5/√(N*M) * exp(-im * phaseK) * inner(psi0, psi_Sja_0)
                phaseK = dot(Q_list[q], (r[centerB]-r[j]))
                chi_t0[m, q] += 0.5/√(N*M) * exp(-im * phaseK) * inner(psi0, psi_Sjb_0)
            end
        end
    end

    # write data
    for q = 1:Nk
        open(filenames[q], "w") do io 
            write(io, "t")
            for m = 1:M 
                write(io, ",RS$m,IS$m")
            end
            write(io, "\n")

            d = @sprintf("%.2f", 0)
            write(io, d)
            for m = 1:M 
                d = @sprintf(",%.10f,%.10f", chi_t0[m, q].re, chi_t0[m, q].im)
                write(io, d)
            end
            write(io, "\n")
        end
    end


    # prepare for time evolution
    psi_SA_t, psi_SB_t = copy(psi_Aprime), copy(psi_Bprime)
    psi_Aprime, psi_Bprime = nothing, nothing
    psi_Sja_0, psi_Sjb_0 = nothing, nothing

    psi_SA_t = expand(psi_SA_t, H; 
        alg="global_krylov", krylovdim=3, cutoff=cutoff
    )
    psi_SB_t = expand(psi_SB_t, H; 
        alg="global_krylov", krylovdim=3, cutoff=cutoff
    )
    nsitesA, nsitesB = 1, 1

    for t in 1:Tsteps

        t_start = time()

        ol = (mod(t, 20) == 1) ? 1 : 0
        t1 = Threads.@spawn tdvp(H, -im*dt, psi_SA_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, cutoff=cutoff,
            nsite=nsitesA, outputlevel=ol
        )
        t2 = Threads.@spawn tdvp(H, -im*dt, psi_SB_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, cutoff=cutoff,
            nsite=nsitesB, outputlevel=ol
        )
        psi_SA_t, psi_SB_t = (fetch(t1), fetch(t2))

        bonddimA, bonddimB = maxlinkdim(psi_SA_t), maxlinkdim(psi_SB_t)
        nsitesA = (bonddimA==maxdim) ? 1 : 2
        nsitesB = (bonddimB==maxdim) ? 1 : 2

        sum_time = time()

        chi_t_p = zeros(ComplexF64, M, Nk)
        # sum over sites
        for m = 1:M
            for n = 1:N 
                j = m+(n-1)*M
                C = 0.5/√(N*M) * exp(im * E0 * t*dt) # from U^\dagger (t)

                psi_SA_t_Si = copy(psi_SA_t)
                psi_SA_t_Si[j] = noprime(op(O1, sites[j]) * psi_SA_t_Si[j])
                pAp = C * inner(psi0, psi_SA_t_Si)
                psi_SB_t_Si = copy(psi_SB_t)
                psi_SB_t_Si[j] = noprime(op(O1, sites[j]) * psi_SB_t_Si[j])
                pBp = C * inner(psi0, psi_SB_t_Si)

                for q = 1:Nk
                    phaseK = dot(Q_list[q], (r[centerA]-r[j]))
                    chi_t_p[m, q] += exp(-im * phaseK) * pAp
                    phaseK = dot(Q_list[q], (r[centerB]-r[j]))
                    chi_t_p[m, q] += exp(-im * phaseK) * pBp
                end
            end 
        end

        for q = 1:Nk
            open(filenames[q], "a") do io 
                d = @sprintf("%.2f", t*dt)
                write(io, d)
                for m = 1:M
                    d = @sprintf(",%.10f,%.10f", chi_t_p[m, q].re, chi_t_p[m, q].im)
                    write(io, d)
                end
                write(io, "\n")
            end
        end


        if mod(t, 20) == 1
            println("T step: $(t), Chi($(t*dt)) finished.")
            println(" ")
            # println("Loop Time spent: $(time()-t_start)")
            println("TDVP time = $(sum_time-t_start)")
            println("Sum time = $(time()-sum_time)")
            println("Process peak RSS (MB): ", Sys.maxrss()/1.04E6)
        end

    end

    return 1
end

function chi_t_scan(ox::Val{true}, oy::Val{false}, N, M, O1, O2, Q_list, r, H, E0, psi0, sites, Tsteps, dt, filenames; cutoff=1e-10, maxdim=20, ns=1, centerA=1, centerB=2, phi_t=false)

    N *= 2 # subsites
    Nk = length(Q_list)

    println("PBC in x, OBC in y. Correlation slices in y direction!")
    println("Core A: ", centerA, "Core B: ", centerB)
    println(" ")
    psi_Aprime, psi_Bprime = copy(psi0), copy(psi0)
    psi_Aprime[centerA] = noprime(op(O2, sites[centerA]) * psi0[centerA])
    psi_Bprime[centerB] = noprime(op(O2, sites[centerB]) * psi0[centerB])

    chi_t0 = zeros(ComplexF64, N, Nk)

    # t = 0
    for n = 1:N
        for m = 1:M
            j = m+(n-1)*M
            psi_Sja_0 = copy(psi_Aprime)
            psi_Sja_0[j] = noprime(op(O1, sites[j]) * psi_Sja_0[j])
            psi_Sjb_0 = copy(psi_Bprime)
            psi_Sjb_0[j] = noprime(op(O1, sites[j]) * psi_Sjb_0[j])

            for q = 1:Nk
                phaseK = dot(Q_list[q], (r[centerA]-r[j]))
                chi_t0[n, q] += 0.5/√(N*M) * exp(-im * phaseK) * inner(psi0, psi_Sja_0)
                phaseK = dot(Q_list[q], (r[centerB]-r[j]))
                chi_t0[n, q] += 0.5/√(N*M) * exp(-im * phaseK) * inner(psi0, psi_Sjb_0)
            end
        end
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
    psi_SA_t, psi_SB_t = copy(psi_Aprime), copy(psi_Bprime)
    psi_Aprime, psi_Bprime = nothing, nothing
    psi_Sja_0, psi_Sjb_0 = nothing, nothing

    psi_SA_t = expand(psi_SA_t, H; 
        alg="global_krylov", krylovdim=3, cutoff=cutoff
    )
    psi_SB_t = expand(psi_SB_t, H; 
        alg="global_krylov", krylovdim=3, cutoff=cutoff
    )
    nsitesA, nsitesB = 1, 1

    for t in 1:Tsteps

        t_start = time()

        ol = (mod(t, 20) == 1) ? 1 : 0
        t1 = Threads.@spawn tdvp(H, -im*dt, psi_SA_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, cutoff=cutoff,
            nsite=nsitesA, outputlevel=ol
        )
        t2 = Threads.@spawn tdvp(H, -im*dt, psi_SB_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, cutoff=cutoff,
            nsite=nsitesB, outputlevel=ol
        )
        psi_SA_t, psi_SB_t = (fetch(t1), fetch(t2))

        bonddimA, bonddimB = maxlinkdim(psi_SA_t), maxlinkdim(psi_SB_t)
        nsitesA = (bonddimA==maxdim) ? 1 : 2
        nsitesB = (bonddimB==maxdim) ? 1 : 2

        sum_time = time()

        chi_t_p = zeros(ComplexF64, N, Nk)
        # sum over sites
        for n = 1:N
            for m = 1:M 
                j = m+(n-1)*M
                C = 0.5/√(N*M) * exp(im * E0 * t*dt) # from U^\dagger (t)

                psi_SA_t_Si = copy(psi_SA_t)
                psi_SA_t_Si[j] = noprime(op(O1, sites[j]) * psi_SA_t_Si[j])
                pAp = C * inner(psi0, psi_SA_t_Si)
                psi_SB_t_Si = copy(psi_SB_t)
                psi_SB_t_Si[j] = noprime(op(O1, sites[j]) * psi_SB_t_Si[j])
                pBp = C * inner(psi0, psi_SB_t_Si)

                for q = 1:Nk
                    phaseK = dot(Q_list[q], (r[centerA]-r[j]))
                    chi_t_p[n, q] += exp(-im * phaseK) * pAp
                    phaseK = dot(Q_list[q], (r[centerB]-r[j]))
                    chi_t_p[n, q] += exp(-im * phaseK) * pBp
                end
            end 
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


        if mod(t, 20) == 1
            println("T step: $(t), Chi($(t*dt)) finished.")
            println(" ")
            # println("Loop Time spent: $(time()-t_start)")
            println("TDVP time = $(sum_time-t_start)")
            println("Sum time = $(time()-sum_time)")
            println("Process peak RSS (MB): ", Sys.maxrss()/1.04E6)
        end

    end

    return 1
end

function chi_x_t(O1, O2, Si, Sj, H, E0, psi0, sites, Tsteps, dt, filename; cutoff=1e-10, maxdim=20, ns=1)

    chi_p = zeros(ComplexF64, Tsteps) # store the chi(t)

    N = length(psi0)
    psi_prime = copy(psi0)
    psi_prime[Sj] = noprime(op(O2, sites[Sj]) * psi0[Sj])

    # prepare for time evolution
    psi_Sj_t = copy(psi_prime)

    # t=0
    psi_prime[Si] = noprime(op(O1, sites[Si]) * psi_prime[Si])
    chi_t0 = complex(0.0, 0.0)
    chi_t0 += inner(psi0, psi_prime)
    
    psi_prime = nothing
    
    # write data
    open(filename, "w") do io 
        write(io, "t,RS,IS\n")
        d = @sprintf("%.2f,%.10f,%.10f\n", 0, chi_t0.re, chi_t0.im)
        write(io, d)
    end

    nsites = 2
    ol = 1
    for t in 1:Tsteps

        cal_t = time()

        # # -- trial --
        # if t<3
        #     nsites = 1
        #     psi_Sj_t = expand(psi_Sj_t, H; 
        #         alg="global_krylov", krylovdim=2, cutoff=cutoff
        #     )
        #     psi_Sj_t = tdvp(
        #         H, -im*dt, psi_Sj_t; 
        #         nsweeps=ns, maxdim=maxdim, normalize=false, nsite=nsites, cutoff=cutoff
        #         , outputlevel=ol
        #     )

        #     # psi_Sj_t = psi_Sj_t + apply(-im*dt*H, psi_Sj_t; cutoff)
        #     # println("Taylor exp")
        # else
        #     nsites = 2
        #     psi_Sj_t = tdvp(
        #         H, -im*dt, psi_Sj_t; 
        #         nsweeps=ns, maxdim=maxdim, normalize=false, nsite=nsites, cutoff=cutoff
        #         , outputlevel=ol
        #     )
        # end
        psi_Sj_t = tdvp(
            H, -im*dt, psi_Sj_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, nsite=nsites, cutoff=cutoff
            , outputlevel=ol
        )
        bonddim = maxlinkdim(psi_Sj_t)
        if bonddim >= maxdim 
            nsites = 1
        end

        psi_Sj_t_Si = copy(psi_Sj_t)
        psi_Sj_t_Si[Si] = noprime(op(O1, sites[Si]) * psi_Sj_t_Si[Si])
        chi_p[t] += inner(psi0, psi_Sj_t_Si) * exp(im * E0 * t*dt)

        if (t % 100) == 0.0 
            println("T step: $(t), Chi($(t*dt)) finished.")
            println("Time spent: $(time()-cal_t)")
        end
        
        # write data incase unexpected termination happend
        open(filename, "a") do io 
            d = @sprintf("%.2f,%.10f,%.10f\n", t*dt, chi_p[t].re, chi_p[t].im)
            write(io, d)
        end
    end

    chi_n = reverse(conj(chi_p)) # for operator SS = 1
    chi = append!(chi_n, chi_t0, chi_p) # time grid from -T to +T

    return chi
end

function chi_x_2t(O1, O2, Si, Sj, H, E0, psi0, sites, Tsteps, dt, filename; cutoff=1e-10, maxdim=20, ns=1)

    chi_p = zeros(ComplexF64, Tsteps) # store the chi(t)
    chi_n = zeros(ComplexF64, Tsteps) # store the chi(t)

    N = length(psi0)
    psi_prime = copy(psi0)
    psi_prime[Sj] = noprime(op(O2, sites[Sj]) * psi0[Sj])

    # prepare for time evolution
    psi_Sj_t, psi_Sj_nt = copy(psi_prime), copy(psi_prime)

    # t=0
    psi_prime[Si] = noprime(op(O1, sites[Si]) * psi_prime[Si])
    chi_t0 = complex(0.0, 0.0)
    chi_t0 += inner(psi0, psi_prime)
    
    psi_prime = nothing

    nsitesp, nsitesn = 1, 1
    ol = 1
    for t in 1:Tsteps

        cal_t = time()
        
        if t == 1
            psi_Sj_t = expand(psi_Sj_t, H; 
                alg="global_krylov", krylovdim=3, cutoff=cutoff
            )
            psi_Sj_nt = expand(psi_Sj_nt, H; 
                alg="global_krylov", krylovdim=3, cutoff=cutoff
            )
        end

        t1 = Threads.@spawn tdvp(
            H, -im*dt, psi_Sj_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, nsite=nsitesp, cutoff=cutoff
            , outputlevel=ol
        )
        t2 = Threads.@spawn tdvp(
            H, im*dt, psi_Sj_nt; 
            nsweeps=ns, maxdim=maxdim, normalize=false, nsite=nsitesn, cutoff=cutoff
            , outputlevel=ol
        )
        psi_Sj_t, psi_Sj_nt = (fetch(t1), fetch(t2))
        bonddimp, bonddimn = maxlinkdim(psi_Sj_t), maxlinkdim(psi_Sj_nt)
        nsitesp = (bonddimp>=maxdim) ? 1 : 2
        nsitesn = (bonddimn>=maxdim) ? 1 : 2

        psi_Sj_t_Si = copy(psi_Sj_t)
        psi_Sj_t_Si[Si] = noprime(op(O1, sites[Si]) * psi_Sj_t_Si[Si])
        chi_p[t] += inner(psi0, psi_Sj_t_Si) * exp(im * E0 * t*dt)
        psi_Sj_t_Si = nothing
        
        psi_Sj_t_Si = copy(psi_Sj_nt)
        psi_Sj_t_Si[Si] = noprime(op(O1, sites[Si]) * psi_Sj_t_Si[Si])
        chi_n[t] += inner(psi0, psi_Sj_t_Si) * exp(-im * E0 * t*dt)
        psi_Sj_t_Si = nothing

        if (t % 100) == 0.0 
            println("T step: $(t), Chi($(t*dt)) finished.")
            println("Time spent: $(time()-cal_t)")
        end
        
    end

    chi_n = reverse(chi_n) # for operator SS = 1
    chi = append!(chi_n, chi_t0, chi_p) # time grid from -T to +T

    return chi
end

function chi_t_FT(O1, O2, Q_list, r, H, psi0, sites, Tsteps, dt, filenames; cutoff=1e-10, maxdim=20, ns=1, phi_t=false, centerA=2, centerB=4)
    
    Nk = length(Q_list)

    N = length(psi0)

    psi_Aprime, psi_Bprime, psi_t = copy(psi0), copy(psi0), copy(psi0)
    psi_Aprime[centerA] = noprime(op(O2, sites[centerA]) * psi0[centerA])
    psi_Bprime[centerB] = noprime(op(O2, sites[centerB]) * psi0[centerB])
    
    # t = 0
    chi_t0 = zeros(ComplexF64, Nk)
    for j = 2:2:N
        psi_Sja_0 = copy(psi_Aprime)
        psi_Sja_0[j] = noprime(op(O1, sites[j]) * psi_Sja_0[j])
        pAp = inner(psi0, psi_Sja_0)
        psi_Sjb_0 = copy(psi_Bprime)
        psi_Sjb_0[j] = noprime(op(O1, sites[j]) * psi_Sjb_0[j])
        pBp = inner(psi0, psi_Sjb_0)

        for q = 1:Nk
            phaseK = dot(Q_list[q], (r[centerA]-r[j]))
            chi_t0[q] += 0.5/√(N/2) * exp(-im*phaseK) * pAp
            phaseK = dot(Q_list[q], (r[centerB]-r[j]))
            chi_t0[q] += 0.5/√(N/2) * exp(-im*phaseK) * pBp
        end
    end
    psi_Sja_0, psi_Sjb_0, pAp, pBp = nothing, nothing, nothing, nothing

    # prepare for time evolution
    psi_SA_t, psi_SB_t = copy(psi_Aprime), copy(psi_Bprime)

    psi_Aprime, psi_Bprime = nothing, nothing

    # write data
    if phi_t
        phiphi = expect(psi_t, "Sz")

        for q = 1:Nk
            open(filenames[q], "w") do io 
                # header
                write(io, "t,RS,IS")
                for n in 1:N
                    write(io, ",Sz$n")
                end
                write(io, "\n")
                # Chi
                d = @sprintf("%.2f,%.10f,%.10f", 0, chi_t0[q].re, chi_t0[q].im)
                write(io, d)
                # Sz_n
                for n in 1:N
                    d = @sprintf(",%.8f", phiphi[n])
                    write(io, d)
                end
                write(io, "\n")
            end
        end

    else
        for q = 1:Nk
            open(filenames[q], "w") do io 
                # header
                write(io, "t,RS,IS\n")
                # Chi
                d = @sprintf("%.2f,%.10f,%.10f\n", 0, chi_t0[q].re, chi_t0[q].im)
                write(io, d)
            end
        end

    end

    for t in 1:Tsteps
        t_start = time()

        ol=1
        bonddimA, bonddimB, bonddimt = maxlinkdim(psi_SA_t), maxlinkdim(psi_SB_t), maxlinkdim(psi_t)
        
        # nsitesA = (bonddimA==maxdim) ? 1 : 2
        # nsitesB = (bonddimB==maxdim) ? 1 : 2
        # nsitest = (bonddimt==maxdim) ? 1 : 2
        nsitesA, nsitesB, nsitest = 1, 1, 1
        
        if bonddimA<maxdim
            psi_SA_t = expand(psi_SA_t, H; 
                alg="global_krylov", krylovdim=3, cutoff=cutoff
            )
        end
        if bonddimB<maxdim
            psi_SB_t = expand(psi_SB_t, H; 
                alg="global_krylov", krylovdim=3, cutoff=cutoff
            )
        end
        if bonddimt<maxdim
            psi_t = expand(psi_t, H; 
                alg="global_krylov", krylovdim=3, cutoff=cutoff
            )
        end

        t1 = Threads.@spawn tdvp(
            H, -im*dt, psi_SA_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, nsite=nsitesA, cutoff=cutoff
            , outputlevel=ol
        )
        t2 = Threads.@spawn tdvp(
            H, -im*dt, psi_SB_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, nsite=nsitesB, cutoff=cutoff
        )
        t3 = Threads.@spawn tdvp(
            H, -im*dt, psi_t; 
            nsweeps=ns, maxdim=maxdim, normalize=true, nsite=nsitest, cutoff=cutoff
            , outputlevel=ol
        )
        psi_SA_t, psi_SB_t, psi_t = (fetch(t1), fetch(t2), fetch(t3))

        println("TDVP Time spent: $(time()-t_start)")
        t_sumi = time()

        chi_t = zeros(ComplexF64, Nk)
        for j = 2:2:N
            psi_Sc_t_Sja = copy(psi_SA_t)
            psi_Sc_t_Sja[j] = noprime(op(O1, sites[j]) * psi_Sc_t_Sja[j])
            pAp = 0.5/√(N/2) * inner(psi_t, psi_Sc_t_Sja)
            psi_Sc_t_Sjb = copy(psi_SB_t)
            psi_Sc_t_Sjb[j] = noprime(op(O1, sites[j]) * psi_Sc_t_Sjb[j])
            pBp = 0.5/√(N/2) * inner(psi_t, psi_Sc_t_Sjb)

            for q = 1:Nk
                phaseK = dot(Q_list[q], (r[centerA]-r[j]))
                chi_t[q] += exp(-im*phaseK) * pAp

                phaseK = dot(Q_list[q], (r[centerB]-r[j]))
                chi_t[q] += exp(-im*phaseK) * pBp 
            end
        end
        println("Sum i Time spent: $(time()-t_sumi)")
        
        # write data incase unexpected termination happend
        if phi_t
            phiphi = expect(psi_t, "Sz")

            for q = 1:Nk
                open(filenames[q], "a") do io 
                    # Chi
                    d = @sprintf("%.2f,%.10f,%.10f", t*dt, chi_t[q].re, chi_t[q].im)
                    write(io, d)
                    # Sz_n
                    for n in 1:N
                        d = @sprintf(",%.8f", phiphi[n])
                        write(io, d)
                    end
                    write(io, "\n")
                end
            end
        else
            for q = 1:Nk
                open(filenames[q], "a") do io 
                    # Chi
                    d = @sprintf("%.2f,%.10f,%.10f\n", t*dt, chi_t[q].re, chi_t[q].im)
                    write(io, d)
                end
            end
        end

        println("T step: $(t), Chi($(t*dt)) finished.")
        println("Loop Time spent: $(time()-t_start)")
        println("Process peak RSS (GB): ", Sys.maxrss()/1.04E9)
        GC.gc()

    end

    return 1
end

function chi_2t_FT(O1, O2, Q, r, H, psi0, sites, Tsteps, dt, filename; cutoff=1e-10, maxdim=20, ns=1, phi_t=false)
    
    chi_p = zeros(ComplexF64, Tsteps) # store the chi(t)

    N = length(psi0)
    centerA, centerB = div(N, 2), div(N, 2)+2
    psi_t = copy(psi0)
    psi_Aprime, psi_Bprime = copy(psi0), copy(psi0)
    psi_Aprime[centerA] = noprime(op(O2, sites[centerA]) * psi0[centerA])
    psi_Bprime[centerB] = noprime(op(O2, sites[centerB]) * psi0[centerB])
    
    # t = 0
    chi_t0 = complex(0.0, 0.0)
    for j = 2:2:N
        psi_Sj_0 = copy(psi_Aprime)
        psi_Sj_0[j] = noprime(op(O1, sites[j]) * psi_Sj_0[j])
        phaseK = dot(Q, (r[centerA]-r[j]))
        chi_t0 += 0.5/√(N/2) * exp(-im*phaseK) * inner(psi0, psi_Sj_0)

        psi_Sj_0 = copy(psi_Bprime)
        psi_Sj_0[j] = noprime(op(O1, sites[j]) * psi_Sj_0[j])
        phaseK = dot(Q, (r[centerB]-r[j]))
        chi_t0 += 0.5/√(N/2) * exp(-im*phaseK) * inner(psi0, psi_Sj_0)
    end

    # prepare for time evolution
    psi_SA_t, psi_SB_t = copy(psi_Aprime), copy(psi_Bprime)

    psi_Aprime, psi_Bprime = nothing, nothing

    open(filename, "w") do io 
        write(io, "t,RS,IS\n")
        d = @sprintf("%.2f,%.10f,%.10f\n", 0, chi_t0.re, chi_t0.im)
        write(io, d)
    end

    for t in 1:Tsteps
        t_start = time()

        ol=1
        bonddimA, bonddimB, bonddimt = maxlinkdim(psi_SA_t), maxlinkdim(psi_SB_t), maxlinkdim(psi_t)
        
        nsitesA = (bonddimA>=maxdim) ? 1 : 2
        nsitesB = (bonddimB>=maxdim) ? 1 : 2
        nsitest = (bonddimt>=maxdim) ? 1 : 2

        t1 = Threads.@spawn tdvp(
            H, -im*dt, psi_SA_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, nsite=nsitesA, cutoff=cutoff
            , outputlevel=ol
        )
        t2 = Threads.@spawn tdvp(
            H, -im*dt, psi_SB_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, nsite=nsitesB, cutoff=cutoff
        )
        t3 = Threads.@spawn tdvp(
            H, -im*dt, psi_t; 
            nsweeps=ns, maxdim=maxdim, normalize=true, nsite=nsitest, cutoff=cutoff
            , outputlevel=ol
        )
        psi_SA_t, psi_SB_t, psi_t = (fetch(t1), fetch(t2), fetch(t3))

        println("TDVP Time spent: $(time()-t_start)")

        chi_t = complex(0.0, 0.0)
        for j = 2:2:N
            psi_Sc_t_Sj = copy(psi_SA_t)
            psi_Sc_t_Sj[j] = noprime(op(O1, sites[j]) * psi_Sc_t_Sj[j])

            phaseK = dot(Q, (r[centerA]-r[j]))
            chi_t += 0.5/√(N/2) * exp(-im*phaseK) * inner(psi_t, psi_Sc_t_Sj)

            psi_Sc_t_Sj = copy(psi_SB_t)
            psi_Sc_t_Sj[j] = noprime(op(O1, sites[j]) * psi_Sc_t_Sj[j])

            phaseK = dot(Q, (r[centerB]-r[j]))
            chi_t += 0.5/√(N/2) * exp(-im*phaseK) * inner(psi_t, psi_Sc_t_Sj)
        end
        # println("Sum i Time spent: $(time()-t_sumi)")
        
        chi_p[t] = chi_t 

        println("T step: $(t), Chi($(t*dt)) finished.")
        println("Loop Time spent: $(time()-t_start)")

        open(filename, "a") do io 
            d = @sprintf("%.2f,%.10f,%.10f\n", t*dt, chi_t.re, chi_t.im)
            write(io, d)
        end
    end

    chi_n = reverse(conj(chi_p))
    chi = append!(chi_n, chi_t0, chi_p) # time grid from -T to +T

    return chi
end

function chi_x_t_FT(O1, O2, Si, Sj, H, psi0, sites, Tsteps, dt, filename; cutoff=1e-10, maxdim=20, ns=1)
    
    chi_p = zeros(ComplexF64, Tsteps) # store the chi(t)

    N = length(psi0)
    psi_t = copy(psi0)
    psi_prime = copy(psi0)
    psi_prime[Sj] = noprime(op(O2, sites[Sj]) * psi0[Sj])

    # prepare for time evolution
    psi_Sj_t = copy(psi_prime)

    # t = 0
    psi_prime[Si] = noprime(op(O1, sites[Si]) * psi_prime[Si])
    chi_t0 = complex(0.0, 0.0)
    chi_t0 += inner(psi0, psi_prime)

    psi_prime = nothing

    # write data
    open(filename, "w") do io 
        write(io, "t,RS,IS\n")
        d = @sprintf("%.2f,%.10f,%.10f\n", 0, chi_t0.re, chi_t0.im)
        write(io, d)
    end

    nsitesA, nsitesB = 2, 2
    ol = 1
    overshootA, overshootB = 0, 0
    for t in 1:Tsteps

        t_start = time()

        # switch back to TDVP1 to save time.
        bonddimA, bonddimB = maxlinkdim(psi_Sj_t), maxlinkdim(psi_t)
        if bonddimA>=maxdim
            nsitesA = 1
            # overshootA = 1
        # elseif (bonddimA<maxdim) & (overshootA==0) 
        #     nsitesA = 1
        #     psi_Sj_t = expand(psi_Sj_t, H; 
        #     alg="global_krylov", krylovdim=2, cutoff=cutoff)
        else
            nsitesA = 2
        end

        if bonddimB>=maxdim
            nsitesB = 1
            # overshootB = 1
        # elseif (bonddimB<maxdim) & (overshootB==0) 
        #     nsitesB = 1
        #     psi_t = expand(psi_t, H; 
        #     alg="global_krylov", krylovdim=2, cutoff=cutoff)
        else 
            nsitesB = 2
        end

        t1 = Threads.@spawn tdvp(
            H, -im*dt, psi_Sj_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, nsite=nsitesA, cutoff=cutoff
            , outputlevel=ol
        )
        t2 = Threads.@spawn tdvp(
            H, -im*dt, psi_t; 
            nsweeps=ns, maxdim=maxdim, normalize=true, nsite=nsitesB, cutoff=cutoff
            # , outputlevel=ol
        )

        psi_Sj_t, psi_t = (fetch(t1), fetch(t2))
        
        println("TDVP Time spent: $(time()-t_start)")

        psi_Sj_t_Si = copy(psi_Sj_t)
        psi_Sj_t_Si[Si] = noprime(op(O1, sites[Si]) * psi_Sj_t[Si])
        chi_p[t] += inner(psi_t, psi_Sj_t_Si) # first operator daggered in the function

        println("T step: $(t), Chi($(t*dt)) finished.")
        println("Loop Time spent: $(time()-t_start)")

        # write data incase unexpected termination happend
        open(filename, "a") do io 
            d = @sprintf("%.2f,%.10f,%.10f\n", t*dt, chi_p[t].re, chi_p[t].im)
            write(io, d)
        end
    end

    chi_n = reverse(conj(chi_p))
    chi = append!(chi_n, chi_t0, chi_p) # time grid from -T to +T

    return chi
end
