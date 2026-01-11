using ITensors
using ITensorMPS: MPS, MPO, OpSum, siteinds, dmrg, random_mps, expect
using Printf
using Statistics: mean
using JLD2

include("DSF.jl")

# https://docs.itensor.org/ITensorMPS/stable/examples/MPSandMPO.html

function H_transfer(N, g)

    os = OpSum()
    J1 = -1
    
    for n = 1:(N-1)
        os .+= J1, "Sz", n, "Sz", n+1
        # os .+= J1/2, "S+", n, "S-", n+1
        # os .+= J1/2, "S-", n, "S+", n+1

        os .+= -g, "Sx", n
    end
    os .+= -g, "Sx", N

    h_pin = 1e-6 
    os .+= h_pin, "Sz", 1

    return os 
end

function H_BLBQ(N, theta)
    #=
    Construct Hamiltonian of Bilinear Biquadratic spin-1 model
    return:
        os: operaters
    =#

    os = OpSum()
    J1 = cos(theta)
    J2 = sin(theta)
    # J1 = 1
    # J2 = 0

    for n = 1:(N-1)
        os .+= J1, "Sz", n, "Sz", n+1
        os .+= J1/2, "S+", n, "S-", n+1
        os .+= J1/2, "S-", n, "S+", n+1

        os .+= J2, "Sz", n, "Sz", n+1, "Sz", n, "Sz", n+1
        os .+= J2/2, "Sz", n, "Sz", n+1, "S+", n, "S-", n+1
        os .+= J2/2, "Sz", n, "Sz", n+1, "S-", n, "S+", n+1

        os .+= J2/2, "S+", n, "S-", n+1, "Sz", n, "Sz", n+1
        os .+= J2/4, "S+", n, "S-", n+1, "S+", n, "S-", n+1
        os .+= J2/4, "S+", n, "S-", n+1, "S-", n, "S+", n+1
        
        os .+= J2/2, "S-", n, "S+", n+1, "Sz", n, "Sz", n+1
        os .+= J2/4, "S-", n, "S+", n+1, "S+", n, "S-", n+1
        os .+= J2/4, "S-", n, "S+", n+1, "S-", n, "S+", n+1
    end

    # # PBC
    # os .+= cos(theta), "Sz", N, "Sz", 1
    # os .+= cos(theta)/2, "S+", N, "S-", 1
    # os .+= cos(theta)/2, "S-", N, "S+", 1

    return os
end


let 
    # parameters
    N = 10
    theta = 0.102*pi
    # println(cos(theta), sin(theta))
    # theta = pi 

    linkdim = 10
    nsweeps = 5
    maxdim = [10, 20, 40, 80, 100]
    cutoff = 1E-10
    dtau = 0.05
    Tsteps = 2000

    k = 4/N * pi
    Omega = range(0.0, 8.0, length = 800)

    # sites = siteinds("S=1/2", N; conserve_sz = false)
    # psi_ran = random_mps(sites; linkdims=linkdim)

    sites = siteinds("S=1/2", N, conserve_sz=true, qnname_sz="TotalSz")
    states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    # states = ["Dn" for n in 1:N]
    psi = MPS(ComplexF64, sites, states)

    H = MPO(H_BLBQ(N, theta), sites)
    energy, psi0 = dmrg(H, psi; nsweeps, maxdim, cutoff, outputlevel=1)
    
    # open("./N10_transfer.csv", "w") do io 
    #     write(io, "g,mx\n")
    # end
    # for g in 0:0.01:2
    #     H = MPO(ComplexF64, H_transfer(N, g), sites)
        
    #     # @printf("Psi_0 link dims = %d\n", linkdim)

    #     energy, psi0 = dmrg(H, psi_ran; nsweeps, maxdim, cutoff, outputlevel=0)
    #     # @printf("Final energy per site = %.12f\n", energy/N)
    #     # println("g = $g")
    #     # println(mean(expect(psi0,"Sx")))

    #     open("./N10_transfer.csv", "a") do io 
    #         d = @sprintf("%.5f,%.5f\n", g, mean(expect(psi0,"Sx")))
    #         write(io, d)
    #     end
    # end

    # TEBD
    # time evolution
    gate_p = TrotterGates(sites, theta, dtau)
    gate_n = TrotterGates(sites, theta, -dtau)
    # chi = chi_t(k, psi0, sites, [gate_p, gate_n], dt=dtau, Tsteps=Tsteps)
    X = 4
    chi = chi_x_t(X, X, psi0, sites, [gate_p, gate_n], dt=dtau, Tsteps=Tsteps)
    # TDVP
    # chi = chi_t(k, H, psi0, sites, Tsteps, dtau; nsites=2, cutoff=cutoff, maxdim=30)
    # chi = chi_x_t(X, X, H, psi0, sites, Tsteps, dtau; nsites=1, cutoff=cutoff, maxdim=30)

    # filename = @sprintf(
    #     "./BLBQ_data/Chi_N%i_obc_theta%.2f_lam%.i_k%.2f_T%.i_dt%.2f_zz_E0%.3f_2TDVP.jld2", 
    #     N, theta/pi, linkdim, k/pi, Tsteps*dtau, dtau, energy
    # )
    filename = @sprintf(
        "./BLBQ_data/Chi_N%i_obc_theta%.3f_lam%.i_x%i_T%.i_dt%.2f_zz_E0%.3f.jld2", 
        N, theta/pi, linkdim, X, Tsteps*dtau, dtau, energy
    )
    save_object(filename, chi)

    # chi = load_object("./BLBQ_data/Chi_N10_theta-0.10_k1.0_T50_dt0.05_zz_E0-16.726.jld2")
    # energy = -16.726
    S = DSF(chi, Omega, energy, Tsteps, dt=dtau)

    # println("DSF=$(S)")    
    # filename = @sprintf(
    #     "./BLBQ_data/S_N%i_obc_theta%.2f_k%.2f_T%.i_dt%.2f_zz_2TDVP.csv", 
    #     N, theta/pi, k/pi, Tsteps*dtau, dtau
    # )
    filename = @sprintf(
        "./BLBQ_data/S_N%i_obc_theta%.3f_x%i_T%.i_dt%.2f_zz.csv", 
        N, theta/pi, X, Tsteps*dtau, dtau
    )
    open(filename, "w") do io 

        write(io, "RS,IS\n")

        for i in 1:length(S) 
            d = @sprintf("%.10f,%.10f\n", S[i].re, S[i].im)
            write(io, d)
        end
    end

end
