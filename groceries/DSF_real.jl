using ITensors, ITensorMPS
using Printf
using JLD2

# https://docs.itensor.org/ITensorMPS/stable/examples/MPSandMPO.html

function H_BLBQ(N, theta)
    #=
    Construct Hamiltonian of Bilinear Biquadratic spin-1 model
    return:
        os: operaters
    =#

    os = OpSum()

    for n = 1:(N-1)
        os .+= cos(theta), "Sz", n, "Sz", n+1
        os .+= cos(theta)/2, "S+", n, "S-", n+1
        os .+= cos(theta)/2, "S-", n, "S+", n+1

        os .+= sin(theta), "Sz", n, "Sz", n+1, "Sz", n, "Sz", n+1
        os .+= sin(theta)/2, "S+", n, "S-", n+1, "S+", n, "S-", n+1
        os .+= sin(theta)/2, "S-", n, "S+", n+1, "S-", n, "S+", n+1
    end

    # # PBC
    # os .+= cos(theta), "Sz", N, "Sz", 1
    # os .+= cos(theta)/2, "S+", N, "S-", 1
    # os .+= cos(theta)/2, "S-", N, "S+", 1

    # os .+= sin(theta), "Sz", N, "Sz", 1, "Sz", N, "Sz", 1
    # os .+= sin(theta)/2, "S+", N, "S-", 1, "S+", N, "S-", 1
    # os .+= sin(theta)/2, "S-", N, "S+", 1, "S-", N, "S+", 1

    return os
end

function TrotterGates(sites, theta, tau)
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

function chi_t(n, psi0, sites, Tgate; dt=0.1, Tsteps=10, cutoff=1E-10)
    #= Calculate the chi(t) in real space for DSF.
        Parameters:
            n: center of correlation
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
    # prepare |psi'> = S_c(n) * |psi>
    psi_prime = copy(psi0)
    psi_prime[n] = noprime(op("S-", sites[n]) * psi_prime[n])
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
        # # sum over sites
        # for j = 1:N
        #     psi_Sj_p = copy(psi_t_p)
        #     psi_Sj_n = copy(psi_t_n)
        #     psi_Sj_p[j] = noprime(op("S+", sites[j]) * psi_Sj_p[j])
        #     psi_Sj_n[j] = noprime(op("S+", sites[j]) * psi_Sj_n[j])

        #     chi_t_p += inner(psi0, psi_Sj_p)
        #     chi_t_n += inner(psi0, psi_Sj_n)
        # end
        
        psi_Sj_p = copy(psi_t_p)
        psi_Sj_n = copy(psi_t_n)
        psi_Sj_p[n] = noprime(op("S+", sites[n]) * psi_Sj_p[n])
        psi_Sj_n[n] = noprime(op("S+", sites[n]) * psi_Sj_n[n])

        chi_t_p += inner(psi0, psi_Sj_p)
        chi_t_n += inner(psi0, psi_Sj_n)
        

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
    psi_Sj_0 = copy(psi_prime)
    # for j = 1:N
    #     psi_Sj_0[j] = noprime(op("S+", sites[j]) * psi_prime[j])
    #     chi_t0 += inner(psi0, psi_Sj_0)
    # end
    psi_Sj_0[n] = noprime(op("S+", sites[n]) * psi_prime[n])
    chi_t0 += inner(psi0, psi_Sj_0)

    chi_n = reverse(chi_n)
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
            integral_sum += dt * exp(im * (omega+E0) * t*dt) * chi[i]
        end
        
        S[idx] = integral_sum
    end

    return S 
end

let 
    # parameters
    N = 10
    theta = pi/12

    linkdim = 20
    nsweeps = 5
    maxdim = [1000, 1500, 2000, 2000, 2500]
    cutoff = 1E-10
    dtau = 0.05
    Tsteps = 1000

    n = 5
    Omega = range(0.0, 8.0, length = 800)

    sites = siteinds("S=1", N)

    H = MPO(H_BLBQ(N, theta), sites)
    
    psi_ran = random_mps(sites; linkdims=linkdim)

    @printf("Psi_0 link dims = %d\n", linkdim)

    energy, psi0 = dmrg(H, psi_ran; nsweeps, maxdim, cutoff)
    @printf("Final energy per site = %.12f\n", energy/N)

    # time evolution
    gate_p = TrotterGates(sites, theta, dtau)
    gate_n = TrotterGates(sites, theta, -dtau)
    chi = chi_t(n, psi0, sites, [gate_p, gate_n], dt=dtau, Tsteps=Tsteps)
    filename = @sprintf(
        "./BLBQ_data/Chir_N%i_theta%.2f_lam%.i_n%.i_T%.i_dt%.2f_pm_E0%.3f.jld2", 
        N, theta/pi, linkdim, n, Tsteps*dtau, dtau, energy
    )
    save_object(filename, chi)

    # chi = load_object("./BLBQ_data/Chi_N10_theta-0.10_k1.0_T50_dt0.05_zz_E0-16.726.jld2")
    # energy = -16.726
    S = DSF(chi, Omega, energy, Tsteps, dt=dtau)

    # println("DSF=$(S)")    
    filename = @sprintf(
        "./BLBQ_data/Sr_N%i_theta%.2f_n%.i_T%.i_dt%.2f_pm.csv", 
        N, theta/pi, n, Tsteps*dtau, dtau
    )
    open(filename, "w") do io 

        header = @sprintf("n%.i\n", n)
        write(io, header)

        for i in 1:length(S) 
            d = @sprintf("%.10f\n", S[i].re)
            write(io, d)
        end
    end

end


#=
After sweep 1 energy=-30.703970400240287  maxlinkdim=90 maxerr=9.78E-11 time=14.081
After sweep 2 energy=-30.89292579009563  maxlinkdim=150 maxerr=1.52E-10 time=3.804
After sweep 3 energy=-30.950192213824213  maxlinkdim=156 maxerr=1.00E-10 time=3.922
After sweep 4 energy=-30.972471168804812  maxlinkdim=152 maxerr=9.97E-11 time=4.560
After sweep 5 energy=-30.982684159224867  maxlinkdim=145 maxerr=1.00E-10 time=3.883
Final energy per site = -0.968208879976
sum to time 0.5
  4.373037 seconds (321.40 k allocations: 4.819 GiB, 13.68% gc time)
sum to time 1.5
120.500944 seconds (3.90 M allocations: 99.006 GiB, 15.85% gc time)
=#
