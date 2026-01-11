using ITensors
using ITensorMPS
using JLD2
using Printf

using LinearAlgebra

println("number of threads: ", Threads.nthreads())

function H_heisenberg_aucillary(N)

    os = OpSum()
    J1 = 1

    for n = 1:2:(N-1-2)
        os .+= J1, "Sz", n, "Sz", n+2
        os .+= J1/2, "S+", n, "S-", n+2
        os .+= J1/2, "S-", n, "S+", n+2
    end

    # PBC
    # os .+= J1, "Sz", N-1, "Sz", 1
    # os .+= J1/2, "S+", N-1, "S-", 1
    # os .+= J1/2, "S-", N-1, "S+", 1

    return os
end

function trivial_state(N)

    # sites = siteinds("S=1/2", N; conserve_sz = true, qnname_sz="TotalSz")
    sites = siteinds("S=1/2", N)

    # Initialize MPS in a simple product state |Up, Up, Up... >
    psi = MPS(sites, "Up")

    gates = ITensor[]
    for j in 1:2:N-1
        s1 = siteind(psi, j)
        s2 = siteind(psi, j+1)
        
        g = ITensor(s1, s2, s1', s2')
        g[s1=>1, s2=>1, s1'=>1, s2'=>1] = 1.0 / sqrt(2)
        g[s1=>1, s2=>1, s1'=>2, s2'=>2] = 1.0 / sqrt(2)

        push!(gates, g)
    end

    psi = apply(gates, psi; cutoff=1e-10)
    psi = noprime(psi)
    println("EPR state fin.")

    return psi, sites
end

function chi_t_FT(k, H, psi0, sites, Tsteps, dt; nsites=1, cutoff=1e-10, maxdim=20, ns=1)

    # Avoid oversubscription
    BLAS.set_num_threads(1)
    chi_p = zeros(ComplexF64, Tsteps) # store the chi(t)
    chi_n = zeros(ComplexF64, Tsteps) # store the chi(t)

    N = length(psi0)
    n = div(N, 2)
    println(N, n)
    psi0_t_p = copy(psi0)
    psi0_t_n = copy(psi0)
    psi_prime = copy(psi0)
    psi_prime[n] = noprime(op("Sz", sites[n]) * psi0[n])
    
    # t = 0
    chi_t0 = complex(0.0, 0.0)
    for j = 1:2:(N-1)
        psi_Sj_0 = copy(psi_prime)
        psi_Sj_0[j] = noprime(op("Sz", sites[j]) * psi_Sj_0[j])
        phaseK = exp(-im * k * (j-n)/2)
        chi_t0 += phaseK * inner(psi0, psi_Sj_0)
    end

    # prepare for time evolution
    psi_t_p = copy(psi_prime)
    psi_t_n = copy(psi_prime)

    psi_prime = nothing

    t_start = time()

    for t in 1:Tsteps
        ol=1

        t_tdvp = time()

        t1 = Threads.@spawn tdvp(
            H, -im*dt, psi_t_p; 
            nsweeps=ns, maxdim=maxdim, normalize=true, nsite=nsites, cutoff=cutoff
            , outputlevel=ol
        )
        t2 = Threads.@spawn tdvp(
            H, im*dt, psi_t_n; 
            nsweeps=ns, maxdim=maxdim, normalize=true, nsite=nsites, cutoff=cutoff
        )
        t3 = Threads.@spawn tdvp(
            H, -im*dt, psi0_t_p; 
            nsweeps=ns, maxdim=maxdim, normalize=true, nsite=nsites, cutoff=cutoff
            , outputlevel=ol
        )
        t4 = Threads.@spawn tdvp(
            H, im*dt, psi0_t_n; 
            nsweeps=ns, maxdim=maxdim, normalize=true, nsite=nsites, cutoff=cutoff
        )

        psi_t_p, psi_t_n, psi0_t_p, psi0_t_n = (fetch(t1), fetch(t2), fetch(t3), fetch(t4))

        println("TDVP Time spent: $(time()-t_tdvp)")
        # GC.gc()
        # println("Live memory: ", Base.summarysize(Main) / 1e6, " MB")

        t_sumi = time()

        chi_t_p = complex(0.0, 0.0)
        chi_t_n = complex(0.0, 0.0)
        # sum over sites
        for j = 1:2:(N-1)
            # println(j)
            psi_Sj_p = copy(psi_t_p)
            psi_Sj_n = copy(psi_t_n)
            psi_Sj_p[j] = noprime(op("Sz", sites[j]) * psi_Sj_p[j])
            psi_Sj_n[j] = noprime(op("Sz", sites[j]) * psi_Sj_n[j])
            # momentum phase
            phaseK = exp(-im * k * (j-n)/2)
            println((j-n)/2)
            chi_t_p += phaseK * inner(psi0_t_p, psi_Sj_p) # first operator daggered in the function
            chi_t_n += phaseK * inner(psi0_t_n, psi_Sj_n)
        end
        println("Sum i Time spent: $(time()-t_sumi)")
        
        chi_p[t] = chi_t_p 
        chi_n[t] = chi_t_n

        println("T step: $(t), Chi($(t*dt)) finished.")
        println("Loop Time spent: $(time()-t_start)")
        t_start = time()
    end

    chi_n = reverse(chi_n)
    chi = append!(chi_n, chi_t0, chi_p) # time grid from -T to +T

    return chi
end

function DSF(chi::AbstractArray{ComplexF64}, Omega::AbstractArray{<:Real}, E0, Tsteps; dt=0.1)
    S = zeros(ComplexF64, length(Omega)) # DSF
    # Perform the discrete Fourier Transform (time integral) for each frequency
    for (idx, omega) in enumerate(Omega)
        integral_sum = complex(0.0)
        for t in -Tsteps:Tsteps
            i = t + Tsteps+1
            # The integral is S = dt * sum_n exp(i*omega*T) * chi(t)
            integral_sum += exp(im * (omega+E0) * t*dt) * chi[i]
        end
        S[idx] = dt * integral_sum
    end
    return S 
end

let
    N_physics = 9
    N = 2 * N_physics
    
    linkdim = 10
    cutoff = 1E-10
    dtau = 0.2
    tausweep = 5
    Tsteps = 10
    maxdim = 40

    beta = 0.01
    d_beta = 0.001
    nsweeps = round(Int, beta/d_beta)
    
    k = pi
    Omega = range(0.0, 5.0, length = 500)
    
    psi_0, sites = trivial_state(N) # |psi(beta=0)>
    H = MPO(H_heisenberg_aucillary(N), sites)

    psi_beta = tdvp(
        H, -beta/2, psi_0; 
        nsweeps=nsweeps, maxdim=20, normalize=true, nsite=2, cutoff=cutoff
        , outputlevel=1
    )
    println("Cooldown fin.")
    println("Beta = ", beta)

    chi = chi_t_FT(k, H, psi_beta, sites, Tsteps, dtau;
        nsites=2, cutoff=1e-10, maxdim=maxdim, ns=tausweep)

    filename = @sprintf(
        "./Heisenberg_data/Chi_N%i_k%.2f_Tau%.i_dt%.2f_lam%.i_beta%.2f_zz.jld2", 
        N_physics, k/pi, Tsteps*dtau, dtau/tausweep, maxdim, beta
    )
    save_object(filename, chi)
    
    S = DSF(chi, Omega, 0, Tsteps, dt=dtau)  
    filename = @sprintf(
        "./Heisenberg_data/S_N%i_k%.2f_T%.i_dt%.2f_lam%.i_beta%.2f_zz.csv", 
        N_physics, k/pi, Tsteps*dtau, dtau/tausweep, maxdim, beta
    )

    open(filename, "w") do io 
        write(io, "RS,IS\n")
        for i in 1:length(S) 
            d = @sprintf("%.10f,%.10f\n", S[i].re, S[i].im)
            write(io, d)
        end
    end

end
