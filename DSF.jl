using ITensors
using ITensorMPS: tdvp, op

function TrotterGates_BLBQ(sites, theta, tau)
    # Return Trotter gates exp(-it * h_ij)

    gates = ITensor[]
    for n = 1:(length(sites)-1)

        s1 = sites[n]
        s2 = sites[n+1]

        h_bil = 
            op("Sz", s1) * op("Sz", s2) + 
            1/2 * op("S+", s1) * op("S-", s2) + 
            1/2 * op("S-", s1) * op("S+", s2)
        h_biq = prime(h_bil) * h_bil
        h_biq = mapprime(h_biq, 2, 1)

        hn = cos(theta)*h_bil + sin(theta)*h_biq

        Gn = exp(-im * tau/2 * hn)
        push!(gates, Gn)
    end

    # # PBC
    # s1 = sites[length(sites)]
    # s2 = sites[1]

    # h_bil = 
    #     op("Sz", s1) * op("Sz", s2) + 
    #     1/2 * op("S+", s1) * op("S-", s2) + 
    #     1/2 * op("S-", s1) * op("S+", s2)
    # h_biq = prime(h_bil) * h_bil
    # h_biq = mapprime(h_biq, 2, 1)

    # hn = cos(theta)*h_bil + sin(theta)*h_biq

    # Gn = exp(-im * tau/2 * hn)
    # push!(gates, Gn)

    append!(gates, reverse(gates))

    return gates
end 

function TrotterGates_Hei(sites, tau)
    # Return Trotter gates exp(-it * h_ij)

    gates = ITensor[]
    for n = 1:(length(sites)-1)

        s1 = sites[n]
        s2 = sites[n+1]

        h = -op("Sz", s1) * op("Sz", s2) 
            -1/2 * op("S+", s1) * op("S-", s2)
            -1/2 * op("S-", s1) * op("S+", s2)

        Gn = exp(-im * tau/2 * h)
        push!(gates, Gn)
    end

    append!(gates, reverse(gates))

    return gates
end 

function chi_t(k, psi0, sites, Tgate; dt=0.1, Tsteps=10, cutoff=1E-10)
    #= Calculate the chi(t) for DSF using TEBD.
        Parameters:
            k: momentum
            psi0: initial wavefunction (MPS)
            sites: site indices
            Tgate: Trotter gates for time evolution, [t+, t-]
            dt: time step
            Tsteps: number of time steps in positive direction. Total length: 2*Tsteps+1
            cutoff: cutoff for SVD in apply function
        Return:
            chi: correlation function chi(t)
    =#

    N = length(psi0)
    chi_p = zeros(ComplexF64, Tsteps) # store the chi(t)
    chi_n = zeros(ComplexF64, Tsteps) # store the chi(t)
    # prepare |psi'> = S_c(0) * |psi>
    center = div(N, 2) + 1
    psi_prime = copy(psi0)
    psi_prime[center] = noprime(op("Sz", sites[center]) * psi_prime[center])
    # normalize!(psi_prime) 

    # prepare for time evolution
    psi_t_p = copy(psi_prime)
    psi_t_n = copy(psi_prime)

    cal_t = time()

    for t in 1:Tsteps

        # update psi_t. After every loop, evolve by U(dt)
        psi_t_p = apply(Tgate[1], psi_t_p; cutoff)
        psi_t_n = apply(Tgate[2], psi_t_n; cutoff)

        chi_t_p = complex(0.0, 0.0)
        chi_t_n = complex(0.0, 0.0)
        # sum over sites
        for j = 1:N
            # println(j)
            psi_Sj_p = copy(psi_t_p)
            psi_Sj_n = copy(psi_t_n)
            psi_Sj_p[j] = noprime(op("Sz", sites[j]) * psi_Sj_p[j])
            psi_Sj_n[j] = noprime(op("Sz", sites[j]) * psi_Sj_n[j])
            # momentum phase
            phaseK = exp(-im * k * (j-center))
            chi_t_p += phaseK * inner(psi0, psi_Sj_p)
            chi_t_n += phaseK * inner(psi0, psi_Sj_n)
        end

        chi_p[t] = chi_t_p 
        chi_n[t] = chi_t_n
        if (t % 100) == 0.0 
            println("T step: $(t), Chi($(t*dt)) finished.")
            println("Time spent: $(time()-cal_t)")
            cal_t = time()
        end
    end 

    # t = 0
    chi_t0 = complex(0.0, 0.0)
    for j = 1:N
        psi_Sj_0 = copy(psi_prime)
        psi_Sj_0[j] = noprime(op("Sz", sites[j]) * psi_prime[j])
        phaseK = exp(-im * k * (j-center))
        chi_t0 += phaseK * inner(psi0, psi_Sj_0)
    end

    chi_n = reverse(chi_n)
    chi = append!(chi_n, chi_t0, chi_p) # time grid from -T to +T

    return chi
end

function chi_x_t(Si, Sj, psi0, E0, sites, Tgates; dt=0.1, Tsteps=10, cutoff=1E-10)
    #= Calculate the chi(t) for DSF using TEBD.
        Parameters:
            Si, Sj: <Si Sj>
            psi0: initial wavefunction (MPS)
            sites: site indices
            Tgate: Trotter gates for time evolution, [t+, t-]
            dt: time step
            Tsteps: number of time steps in positive direction. Total length: 2*Tsteps+1
            cutoff: cutoff for SVD in apply function
        Return:
            chi: correlation function chi(t)
    =#

    N = length(psi0)
    chi_p = zeros(ComplexF64, Tsteps) # store the chi(t)
    chi_n = zeros(ComplexF64, Tsteps) # store the chi(t)
    # prepare |psi'> = Sj(0) * |psi>
    psi_prime = copy(psi0)
    psi0_t_p = copy(psi0)
    psi0_t_n = copy(psi0)
    psi_prime[Sj] = noprime(op("Sz", sites[Sj]) * psi_prime[Sj])
    # normalize!(psi_prime) 

    # prepare for time evolution
    psi_t_p = copy(psi_prime)
    psi_t_n = copy(psi_prime)

    cal_t = time()

    for t in 1:Tsteps

        # update psi_t. After every loop, evolve by U(dt)
        psi_t_p = apply(Tgates[1], psi_t_p; cutoff)
        psi_t_n = apply(Tgates[2], psi_t_n; cutoff)
        psi0_t_p = apply(Tgates[1], psi0_t_p; cutoff)
        psi0_t_n = apply(Tgates[2], psi0_t_n; cutoff)

        psi_Sj_p = copy(psi_t_p)
        psi_Sj_n = copy(psi_t_n)
        psi_Sj_p[Si] = noprime(op("Sz", sites[Si]) * psi_Sj_p[Si])
        psi_Sj_n[Si] = noprime(op("Sz", sites[Si]) * psi_Sj_n[Si])

        chi_p[t] = inner(psi0, psi_Sj_p) * exp(im * E0 * t*dt)
        chi_n[t] = inner(psi0, psi_Sj_n) * exp(-im * E0 * t*dt)

        if (t % 100) == 0.0 
            println("T step: $(t), Chi($(t*dt)) finished.")
            println("Time spent: $(time()-cal_t)")
            cal_t = time()
        end
    end 

    # t = 0
    psi_Sj_0 = copy(psi_prime)
    psi_Sj_0[Si] = noprime(op("Sz", sites[Si]) * psi_prime[Si])
    chi_t0 = inner(psi0, psi_Sj_0)

    chi_n = reverse(chi_n)
    chi = append!(chi_n, chi_t0, chi_p) # time grid from -T to +T

    return chi
end

function chi_t(k, H, psi0, sites, Tsteps, dt; nsites=1, cutoff=1e-10, maxdim=20)

    # # observer method (optimization)
    # mutable struct DSFObserver <: AbstractObserver # store observable
    #     times::Vector{Float64}
    #     chi::Vector{ComplexF64}
    # end

    # function ITensors.measure!(obs::DSFObserver; psi, t, kwargs...)
    #     push!(obs.times, imag(t)) # imag(t) if doing real-time with -im*dt
        
    #     corr = inner(psi_0, psi) 
    #     push!(obs.correlations, corr)
    # end

    # manual loop (for verification)

    chi_p = zeros(ComplexF64, Tsteps) # store the chi(t)
    chi_n = zeros(ComplexF64, Tsteps) # store the chi(t)

    N = length(psi0)
    center = div(N, 2) + 1
    psi_prime = copy(psi0)
    psi_prime[center] = noprime(op("Sz", sites[center]) * psi0[center])

    # prepare for time evolution
    psi_t_p = copy(psi_prime)
    psi_t_n = copy(psi_prime)

    cal_t = time()

    for t in 1:Tsteps
        psi_t_p = tdvp(
            H, -im*dt, psi_t_p; 
            nsweeps=1, maxdim=maxdim, normalize=true, nsite=nsites
        )
        psi_t_n = tdvp(
            H, im*dt, psi_t_n; 
            nsweeps=1, maxdim=maxdim, normalize=true, nsite=nsites
        )

        chi_t_p = complex(0.0, 0.0)
        chi_t_n = complex(0.0, 0.0)
        # sum over sites
        for j = 1:N
            # println(j)
            psi_Sj_p = copy(psi_t_p)
            psi_Sj_n = copy(psi_t_n)
            psi_Sj_p[j] = noprime(op("Sz", sites[j]) * psi_Sj_p[j])
            psi_Sj_n[j] = noprime(op("Sz", sites[j]) * psi_Sj_n[j])
            # momentum phase
            phaseK = exp(-im * k * (j-center))
            chi_t_p += phaseK * inner(psi0, psi_Sj_p)
            chi_t_n += phaseK * inner(psi0, psi_Sj_n)
        end
        
        chi_p[t] = chi_t_p 
        chi_n[t] = chi_t_n
        if (t % 100) == 0.0 
            println("T step: $(t), Chi($(t*dt)) finished.")
            println("Time spent: $(time()-cal_t)")
            cal_t = time()
        end
    end

    # t = 0
    chi_t0 = complex(0.0, 0.0)
    for j = 1:N
        psi_Sj_0 = copy(psi_prime)
        psi_Sj_0[j] = noprime(op("Sz", sites[j]) * psi_prime[j])
        phaseK = exp(-im * k * (j-center))
        chi_t0 += phaseK * inner(psi0, psi_Sj_0)
    end

    chi_n = reverse(chi_n)
    chi = append!(chi_n, chi_t0, chi_p) # time grid from -T to +T

    return chi
end

function chi_t_test(k, H, psi0, sites, Tsteps, dt; nsites=1, cutoff=1e-10, maxdim=20)

    # # observer method (optimization)
    # mutable struct DSFObserver <: AbstractObserver # store observable
    #     times::Vector{Float64}
    #     chi::Vector{ComplexF64}
    # end

    # function ITensors.measure!(obs::DSFObserver; psi, t, kwargs...)
    #     push!(obs.times, imag(t)) # imag(t) if doing real-time with -im*dt
        
    #     corr = inner(psi_0, psi) 
    #     push!(obs.correlations, corr)
    # end

    # manual loop (for verification)

    chi_p = zeros(ComplexF64, Tsteps) # store the chi(t)
    chi_n = zeros(ComplexF64, Tsteps) # store the chi(t)

    N = length(psi0)
    center = div(N, 2) + 1
    psi_prime = copy(psi0)
    psi0_t_p = copy(psi0)
    psi0_t_n = copy(psi0)
    psi_prime[center] = noprime(op("Sz", sites[center]) * psi0[center])

    # prepare for time evolution
    psi_t_p = copy(psi_prime)
    psi_t_n = copy(psi_prime)

    cal_t = time()

    for t in 1:Tsteps
        psi_t_p = tdvp(
            H, -im*dt, psi_t_p; 
            nsweeps=1, maxdim=maxdim, normalize=true, nsite=nsites
        )
        psi_t_n = tdvp(
            H, im*dt, psi_t_n; 
            nsweeps=1, maxdim=maxdim, normalize=true, nsite=nsites
        )
        psi0_t_p = tdvp(
            H, -im*dt, psi0_t_p; 
            nsweeps=1, maxdim=maxdim, normalize=true, nsite=nsites
        )
        psi0_t_n = tdvp(
            H, im*dt, psi0_t_n; 
            nsweeps=1, maxdim=maxdim, normalize=true, nsite=nsites
        )

        chi_t_p = complex(0.0, 0.0)
        chi_t_n = complex(0.0, 0.0)
        # sum over sites
        for j = 1:N
            # println(j)
            psi_Sj_p = copy(psi_t_p)
            psi_Sj_n = copy(psi_t_n)
            psi_Sj_p[j] = noprime(op("Sz", sites[j]) * psi_Sj_p[j])
            psi_Sj_n[j] = noprime(op("Sz", sites[j]) * psi_Sj_n[j])
            # momentum phase
            phaseK = exp(-im * k * (j-center))
            chi_t_p += phaseK * inner(psi0_t_p, psi_Sj_p)
            chi_t_n += phaseK * inner(psi0_t_n, psi_Sj_n)
        end
        
        chi_p[t] = chi_t_p 
        chi_n[t] = chi_t_n
        if (t % 100) == 0.0 
            println("T step: $(t), Chi($(t*dt)) finished.")
            println("Time spent: $(time()-cal_t)")
            cal_t = time()
        end
    end

    # t = 0
    chi_t0 = complex(0.0, 0.0)
    for j = 1:N
        psi_Sj_0 = copy(psi_prime)
        psi_Sj_0[j] = noprime(op("Sz", sites[j]) * psi_prime[j])
        phaseK = exp(-im * k * (j-center))
        chi_t0 += phaseK * inner(psi0, psi_Sj_0)
    end

    chi_n = reverse(chi_n)
    chi = append!(chi_n, chi_t0, chi_p) # time grid from -T to +T

    return chi
end

function chi_t_FT(k, H, psi0, sites, Tsteps, dt; nsites=1, cutoff=1e-10, maxdim=20, ns=1)
    
    chi_p = zeros(ComplexF64, Tsteps) # store the chi(t)

    N = length(psi0)
    n = div(N, 2)
    psi_t = copy(psi0)
    psi_prime = copy(psi0)
    psi_prime[n] = noprime(op("Sz", sites[n]) * psi0[n])
    
    # t = 0
    chi_t0 = complex(0.0, 0.0)
    for j = 1:2:(N-1)
        psi_Sj_0 = copy(psi_prime)
        psi_Sj_0[j] = noprime(op("Sz", sites[j]) * psi_prime[j])
        phaseK = exp(-im * k * (j-n)/2)
        chi_t0 += phaseK * inner(psi0, psi_Sj_0)
    end

    # prepare for time evolution
    psi_Sc_t = copy(psi_prime)

    psi_prime = nothing

    for t in 1:Tsteps
        ol=1

        t_start = time()

        t1 = Threads.@spawn tdvp(
            H, -im*dt, psi_Sc_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, nsite=nsites, cutoff=cutoff
            , outputlevel=ol
        )
        t2 = Threads.@spawn tdvp(
            H, -im*dt, psi_t; 
            nsweeps=ns, maxdim=maxdim, normalize=true, nsite=nsites, cutoff=cutoff
            , outputlevel=ol
        )
        psi_Sc_t, psi_t = (fetch(t1), fetch(t2))

        println("TDVP Time spent: $(time()-t_start)")

        t_sumi = time()

        chi_t = complex(0.0, 0.0)
        # sum over sites
        for j = 1:2:(N-1)
            psi_Sc_t_Sj = copy(psi_Sc_t)
            psi_Sc_t_Sj[j] = noprime(op("Sz", sites[j]) * psi_Sc_t_Sj[j])

            # momentum phase
            phaseK = exp(-im * k * (j-n)/2)
            chi_t += phaseK * inner(psi_t, psi_Sc_t_Sj) 
        end
        println("Sum i Time spent: $(time()-t_sumi)")
        
        chi_p[t] = chi_t 

        println("T step: $(t), Chi($(t*dt)) finished.")
        println("Loop Time spent: $(time()-t_start)")
    end

    chi_n = reverse(conj(chi_p))
    chi = append!(chi_n, chi_t0, chi_p) # time grid from -T to +T

    return chi
end

function chi_x_t_FT(Si, Sj, H, psi0, sites, Tsteps, dt; nsites=1, cutoff=1e-10, maxdim=20, ns=1)
    
    chi_p = zeros(ComplexF64, Tsteps) # store the chi(t)

    N = length(psi0)
    psi_t = copy(psi0)
    psi_prime = copy(psi0)
    psi_prime[Sj] = noprime(op("Sz", sites[Sj]) * psi0[Sj])

    # prepare for time evolution
    psi_Sj_t = copy(psi_prime)

    # t = 0
    psi_prime[Si] = noprime(op("Sz", sites[Si]) * psi_prime[Si])
    chi_t0 = inner(psi0, psi_prime)

    psi_prime = nothing

    for t in 1:Tsteps

        ol=1

        t_start = time()

        t1 = Threads.@spawn tdvp(
            H, -im*dt, psi_Sj_t; 
            nsweeps=ns, maxdim=maxdim, normalize=false, nsite=nsites, cutoff=cutoff
            , outputlevel=ol
        )
        t2 = Threads.@spawn tdvp(
            H, -im*dt, psi_t; 
            nsweeps=ns, maxdim=maxdim, normalize=true, nsite=nsites, cutoff=cutoff
            # , outputlevel=ol
        )

        psi_Sj_t, psi_t = (fetch(t1), fetch(t2))

        println("TDVP Time spent: $(time()-t_start)")

        psi_Sj_t_Si = copy(psi_Sj_t)
        psi_Sj_t_Si[Si] = noprime(op("Sz", sites[Si]) * psi_Sj_t[Si])
        chi_p[t] += inner(psi_t, psi_Sj_t_Si) # first operator daggered in the function

        println("T step: $(t), Chi($(t*dt)) finished.")
        println("Loop Time spent: $(time()-t_start)")
    end

    chi_n = reverse(conj(chi_p))
    chi = append!(chi_n, chi_t0, chi_p) # time grid from -T to +T

    return chi
end


function DSF(chi::AbstractArray{ComplexF64}, Omega::AbstractArray{<:Real}, E0, Tsteps; dt=0.1)
    #= Integrate chi(t) to get DSF = 
        sum_t dt sum_j exp[i(omega-E_0)t - ik(r_j(t)-r_c(0))] * <psi| S_j U(t) S_c |psi>.
        Parameters:
        chi: array
            time series of chi from function chi_t
        Omega: array
            desired omega in the unit of J
        Return:
            S: dynamical structure factor
    =#

    S = zeros(ComplexF64, length(Omega)) # DSF

    # Perform the discrete Fourier Transform (time integral) for each frequency
    for (idx, omega) in enumerate(Omega)
        integral_sum = complex(0.0)
        
        for t in -Tsteps:Tsteps
            i = t + Tsteps+1
            # The integral is S = dt * sum_n exp(i*omega*T) * chi(t)
            integral_sum += exp(im * (omega+E0) * t*dt) * chi[i]
        end
        
        # S(k, omega) is proportional to the time integral
        # Since we use dt * sum, this already approximates the integral.
        S[idx] = dt * integral_sum
    end

    return S 
end


function DSF_FT(chi::AbstractArray{ComplexF64}, Omega::AbstractArray{<:Real}, Tsteps; dt=0.1)
    #= Integrate chi(t) to get DSF = 
        sum_t dt sum_j exp[i omega t - ik(r_j(t)-r_c(0))] * <psi|U(-t) S_j U(t) S_c |psi>.
        Parameters:
        chi: array
            time series of chi from function chi_t
        Omega: array
            desired omega in the unit of J
        Return:
            S: dynamical structure factor
    =#

    return DSF(chi, Omega, 0, Tsteps; dt=dt)
end
