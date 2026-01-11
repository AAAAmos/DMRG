using ITensors, ITensorMPS
using Printf

# https://docs.itensor.org/ITensorMPS/stable/examples/MPSandMPO.html

function H_SSH(N, t1, t2)
    #=
    Construct Hamiltonian of SSH spin-1 model?
    return:
        os: operaters
    =#

    os = OpSum()

    for n = 1:N
        os .+= t1, "Sz", 2*n-1, "Sz", 2*n
        os .+= t1/2, "S+", 2*n-1, "S-", 2*n
        os .+= t1/2, "S-", 2*n-1, "S+", 2*n
    end
    
    for n = 1:N-1
        os .+= t2, "Sz", 2*n, "Sz", 2*n+1
        os .+= t2/2, "S+", 2*n, "S-", 2*n+1
        os .+= t2/2, "S-", 2*n, "S+", 2*n+1
    end

    return os
end

function TrotterGates(sites, t1, t2, dtau)

    gates = ITensor[]
    for n = 1:(length(sites)-1)

        sA = sites[2*n-1]
        sB = sites[2*n]
        sA_ = sites[2*n+1]

        hn = 
            t1 * (op("Sz", sA) * op("Sz", sB) +
            1/2 * op("S+", sA) * op("S-", sB) +
            1/2 * op("S-", sA) * op("S+", sB)) 
        Gn = exp(-im * dtau/2 * hn)
        push!(gates, Gn)

        hn = 
            t2 * (op("Sz", sB) * op("Sz", sA_) +
            1/2 * op("S+", sB) * op("S-", sA_) +
            1/2 * op("S-", sB) * op("S+", sA_)) 
        Gn = exp(-im * dtau/2 * hn)
        push!(gates, Gn)
    end
    hn = 
        t1 * (op("Sz", sites[2*N-1]) * op("Sz", sites[2*N]) +
        1/2 * op("S+", sites[2*N-1]) * op("S-", sites[2*N]) +
        1/2 * op("S-", sites[2*N-1]) * op("S+", sites[2*N])) 
    Gn = exp(-im * dtau/2 * hn)
    push!(gates, Gn)

    append!(gates, reverse(gates))

    return gates
end 

function chi_t(k, psi0, sites, Tgate; dt=0.1, Tsteps=10, cutoff=1E-10)

    N = length(psi0)
    chi_p = zeros(ComplexF64, Tsteps)
    chi_n = zeros(ComplexF64, Tsteps)

    center = div(N, 2)
    psi_prime = copy(psi0)
    psi_prime[center] = noprime(op("Sz", sites[center]) * psi_prime[center])
    normalize!(psi_prime)

    psi_t = copy(psi_prime)

    for t in 1:Tsteps
        psi_t_p = apply(Tgate[1], psi_t)
end

function DSF(k, omega, psi0, sites, Tgate; dt=0.1, Tsteps=10, cutoff=1E-10)

    N = length(psi0)
    chi = Array{ComplexF64}(undef, Tsteps) # chi(t)
    S = complex(0, 0) # DSF
    center = div(N, 2)

    # S_c(0) * |psi>
    psi0[center] = noprime(op("S-", sites[center]) * psi0[center])

    function St(steps)
        # Construct sum_j exp[-ik(r_j(t)-r_c(0))] * <S_j S_c> 
        # for single time t = dt * steps

        psi_t = copy(psi0)

        for t = 0.5:steps
            psi_t = apply(Tgate, psi_t; cutoff)
        end

        # chi_i(t)
        chi_i = complex(0, 0)
        # loop over sites
        for i = 1:N
            # S_j(0) U(t) S_c(0) |psi>
            psi_Si = copy(psi_t)
            psi_Si[i] = noprime(op("S+", sites[i]) * psi_Si[i])
            for t = 1:steps 
                psi_Si = apply(conj(Tgate), psi_Si; cutoff)
            end

            # <psi| S_j(t) S_c(0) |psi>
            phaseK = exp(-im * k * (i-center))
            chi_i += phaseK * inner(conj(psi0), psi_Si)
        end

        return chi_i 
    end

    time_St(t) = @time St(t)

    # integral over time
    for t = 0.5:Tsteps-0.5 
        S += dt * exp.(im*omega*dt*t) * St(t)
        println("sum to time ", t)
        time_St(t)
    end
    
    return S
end


let 
    # parameters
    N = 32
    t1, t2 = 1, 2

    linkdim = 10
    nsweeps = 5
    maxdim = [100,150,200,200,250]
    cutoff = 1E-10
    dtau = 0.1
    ttotal = 1.0

    sites = siteinds("S=1/2", 2*N)

    H = MPO(H_SSH(N, t1, t2), sites)
    
    psi0 = random_mps(sites; linkdims=linkdim)

    @printf("Psi_0 link dims = %d\n", linkdim)

    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    @printf("Final energy per site = %.12f\n", energy/(2*N))

    # time evolution
    gates = TrotterGates(N, sites, t1, t2, dtau)
    Omega = (Float32)[0.1, 0.5, 1.0]
    DSF(0, Omega, psi0, sites, gates)

end

#=
Psi_0 link dims = 10
After sweep 1 energy=-48.36186553261613  maxlinkdim=31 maxerr=9.86E-11 time=11.563
After sweep 2 energy=-48.36198431225511  maxlinkdim=17 maxerr=9.99E-11 time=0.144
After sweep 3 energy=-48.361984312698574  maxlinkdim=17 maxerr=9.96E-11 time=0.093
After sweep 4 energy=-48.36198431269896  maxlinkdim=17 maxerr=9.96E-11 time=0.095
After sweep 5 energy=-48.361984312698866  maxlinkdim=17 maxerr=9.96E-11 time=0.092
Final energy per site = -0.755656004886
sum i fin
sum to time 0.5
sum i fin
  0.276943 seconds (1.11 M allocations: 568.399 MiB, 22.66% gc time)
sum i fin
sum to time 1.5
sum i fin
 14.701342 seconds (18.08 M allocations: 15.663 GiB, 16.35% gc time)
sum i fin
sum to time 2.5
sum i fin
 34.174526 seconds (34.23 M allocations: 33.534 GiB, 14.95% gc time)
sum i fin
sum to time 3.5
sum i fin
 58.637183 seconds (50.39 M allocations: 55.153 GiB, 15.62% gc time)
sum i fin
sum to time 4.5
sum i fin
 90.026522 seconds (66.55 M allocations: 82.297 GiB, 15.06% gc time)
sum i fin
sum to time 5.5
sum i fin
123.707439 seconds (82.70 M allocations: 116.807 GiB, 15.16% gc time)
sum i fin
sum to time 6.5
sum i fin
165.783367 seconds (98.87 M allocations: 159.535 GiB, 15.12% gc time)
sum i fin
sum to time 7.5
sum i fin
218.825527 seconds (115.03 M allocations: 212.397 GiB, 14.87% gc time)
sum i fin
sum to time 8.5
sum i fin
287.049053 seconds (131.20 M allocations: 276.426 GiB, 14.27% gc time)
sum i fin
sum to time 9.5
sum i fin
368.272622 seconds (147.36 M allocations: 352.355 GiB, 14.28% gc time)
=#