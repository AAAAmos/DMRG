using ITensors
using ITensorMPS
using Printf: @sprintf
using LinearAlgebra: BLAS
using Statistics: mean
# using JLD2

include("DSF.jl")

# keep the ancillary space till the end?
# can I conserve quantum number?

function H_heisenberg(N)

    os = OpSum()
    J1 = 1

    for n = 1:(N-1)
        os .+= J1, "Sz", n, "Sz", n+1
        os .+= J1/2, "S+", n, "S-", n+1
        os .+= J1/2, "S-", n, "S+", n+1
    end

    # PBC
    # os .+= J1, "Sz", N, "Sz", 1
    # os .+= J1/2, "S+", N, "S-", 1
    # os .+= J1/2, "S-", N, "S+", 1

    # h_pin = 1e-6 
    # os .+= h_pin, "Sz", 1

    return os
end

function H_heisenberg_aucillary(N)

    os = OpSum()
    J1 = 1
    # g = 1

    for n = 1:2:(N-1-2)
        os .+= J1, "Sz", n, "Sz", n+2
        os .+= J1/2, "S+", n, "S-", n+2
        os .+= J1/2, "S-", n, "S+", n+2
        # os .+= g, "Sz", n
    end
    # os .+= g, "Sz", N-1

    # PBC
    # os .+= J1, "Sz", N-1, "Sz", 1
    # os .+= J1/2, "S+", N-1, "S-", 1
    # os .+= J1/2, "S-", N-1, "S+", 1

    return os
end


function H_BLBQ_aucillary(N, theta)
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

    for n = 2:2:(N-2)
        os .+= J1, "Sz", n, "Sz", n+2
        os .+= J1/2, "S+", n, "S-", n+2
        os .+= J1/2, "S-", n, "S+", n+2

        os .+= J2, "Sz", n, "Sz", n+2, "Sz", n, "Sz", n+2
        os .+= J2/2, "Sz", n, "Sz", n+2, "S+", n, "S-", n+2
        os .+= J2/2, "Sz", n, "Sz", n+2, "S-", n, "S+", n+2

        os .+= J2/2, "S+", n, "S-", n+2, "Sz", n, "Sz", n+2
        os .+= J2/4, "S+", n, "S-", n+2, "S+", n, "S-", n+2
        os .+= J2/4, "S+", n, "S-", n+2, "S-", n, "S+", n+2
        
        os .+= J2/2, "S-", n, "S+", n+2, "Sz", n, "Sz", n+2
        os .+= J2/4, "S-", n, "S+", n+2, "S+", n, "S-", n+2
        os .+= J2/4, "S-", n, "S+", n+2, "S-", n, "S+", n+2
    end

    # # PBC
    # os .+= cos(theta), "Sz", N, "Sz", 1
    # os .+= cos(theta)/2, "S+", N, "S-", 1
    # os .+= cos(theta)/2, "S-", N, "S+", 1

    return os
end

let
    N_physics = 9
    N = 2 * N_physics
    QN_conservation = false
    
    linkdim = 20
    dmrg_maxdim = [200, 200, 200, 200, 200]
    psi_cutoff = 1E-8
    time_cutoff = 1E-10
    println("State cutoff: ", psi_cutoff)
    dtau = 0.08
    tausweep = 1
    Tsteps = 15*5

    theta = 0.102*pi # AKLT phase

    beta = 1
    beta_list = [0, 1]
    # beta_list = [0, 1/100, 2/100, 1/10, 2/10, 5/10, 1, 2, 3, 4, 5, 7]
    # beta_list = [0, 1, 2, 3, 4, 5, 7]
    d_beta = 0.01
    # E_file = "./Heisenberg_data/v1/N9_E_cut2.5-9_dbeta0.001.csv"
    
    k = pi
    Omega = range(0.0, 5.0, length = 500)
    
#    --- T=0 ---

    # -- physical states --
    # sites = siteinds("S=1/2", N_physics; conserve_sz = true)
    # states = [isodd(n) ? "Up" : "Dn" for n in 1:N_physics] # AFM
    # # states = ["Dn" for n in 1:N_physics]
    # # states = ["Up" for n in 1:N_physics]
    # psi = MPS(Float64, sites, states)

    # # sites = siteinds("S=1/2", N; conserve_sz = false)
    # # psi_ran = random_mps(sites; linkdims=linkdim)

    # H = MPO(H_heisenberg(N_physics), sites)

#    -- dmrg --
    # E0, psi0 = dmrg(H, psi; 
    #     nsweeps=5, maxdim=dmrg_maxdim, cutoff=psi_cutoff, outputlevel=1
    # )
    # mean_sz = mean(expect(psi0, "Sz"))
    # # println("Sz(5)=", expect(psi0, "Sz")[5])
    # println("Ground state energy: ", E0)

    # open(E_file, "w") do io 
    #     write(io, "beta,E,Sz,dim\n")
    #     d = @sprintf("%.i,%.10f,%.5f,%.i\n", 100000, E0, mean_sz, maxlinkdim(psi0))
    #     write(io, d)
    # end

#    -- correlation function --
#    -- real space --
    # Si = 1
    # Sj = Si

    # filename = @sprintf(
    #     "./Heisenberg_data/v2/Chi_obc_N%i_S%iS%i_Tau%i_dt%.2f_GS_zz.csv", 
    #     # "./test.csv", 
    #     N_physics, Si, Sj, Tsteps*dtau, dtau/tausweep
    # )

    # chi = chi_x_t(Si, Sj, H, psi0, sites, Tsteps, dtau, filename;
    #     cutoff=time_cutoff, maxdim=50, ns=tausweep
    # )

    # # Tgatep = TrotterGates_Hei(sites, dtau)
    # # Tgaten = TrotterGates_Hei(sites, -dtau)
    # # chi = chi_x_t(Si, Si, psi0, E0, sites, [Tgatep, Tgaten]; dt=dtau, Tsteps=Tsteps, cutoff=1E-10)

    # open(filename, "w") do io 
    #     write(io, "t,RS,IS\n")
    #     for t in -Tsteps:Tsteps
    #         i = t + Tsteps+1
    #         d = @sprintf("%.2f,%.10f,%.10f\n", t*dtau, chi[i].re, chi[i].im)
    #         write(io, d)
    #     end
    # end

#    --- finite T ---
    psi_beta, sites = trivial_state(N; QN=QN_conservation) # |psi(beta=0)>
    H = MPO(H_heisenberg_aucillary(N), sites)
    
    # println("Purity: ", purity(psi_beta, sites))

    for i in 1:length(beta_list)-1

        b = beta_list[i+1]-beta_list[i]
        nsweeps = round(Int, b/d_beta)
        println("delta beta:", d_beta)
        # println("number of loops:", nsweeps)

        psi_beta = tdvp(
            H, -b/2, psi_beta; 
            nsweeps=nsweeps, maxdim=1000, normalize=true, nsite=2, cutoff=psi_cutoff
            , outputlevel=1
        )
        println("Cooldown fin.")
        beta = beta_list[i+1]
        println("Beta = ", beta)
    end

#    - Mz, Trace, E -
    #     Mz = mean(expect(psi_beta, "Sz"))
    #     dim = maxlinkdim(psi_beta)

    #     # site number < 14 is needed !!!
    #     # println("Purity: ", purity(psi_beta, sites))

    #     psi_h = apply(H, psi_beta; cutoff=psi_cutoff)
    #     E_beta = inner(psi_beta, psi_h)
    #     # println("Energy: ", E_beta)
    #     # println("M: ", Mz)
    #     open(E_file, "a") do io 
    #         d = @sprintf("%.4f,%.10f,%.5f,%.i\n", beta, E_beta, Mz, dim)
    #         write(io, d)
    #     end
    # end

#    -- Real space, time --
    BLAS.set_num_threads(1)

    Si = 9
    Sj = Si

    filename = @sprintf(
        "./Heisenberg_data/v2/Chi_N%i_S%iS%i_Tau%i_dt%.2f_beta%.2f_zz_test_ex1+1_dim.csv", 
        N_physics, Si-4, Sj-4, Tsteps*dtau, dtau/tausweep, beta
    )

    chi = chi_x_t_FT(Si, Sj, H, psi_beta, sites, Tsteps, dtau, filename;
        cutoff=time_cutoff, maxdim=50, ns=tausweep
    )

    # open(filename, "w") do io 
    #     write(io, "t,RS,IS\n")
    #     for t in -Tsteps:Tsteps
    #         i = t + Tsteps+1
    #         d = @sprintf("%.2f,%.10f,%.10f\n", t*dtau, chi[i].re, chi[i].im)
    #         write(io, d)
    #     end
    # end

#    -- Momentum space, real time --
    # BLAS.set_num_threads(1)
    
    # filename = @sprintf(
    #     "./Heisenberg_data/Chi_N%i_k%.2f_Tau%.i_dt%.2f_beta%.2f_zz_err-6.csv", 
    #     N_physics, k/pi, Tsteps*dtau, dtau/tausweep, beta
    # )

    # chi = chi_t_FT(k, H, psi_beta, sites, Tsteps, dtau, filename; 
    #     nsites=2, cutoff=psi_cutoff, maxdim=1000, ns=tausweep
    # )
    # open(filename, "w") do io 
    #     write(io, "t,RS,IS\n")
    #     for t in -Tsteps:Tsteps
    #         i = t + Tsteps+1
    #         d = @sprintf("%.2f,%.10f,%.10f\n", t*dtau, chi[i].re, chi[i].im)
    #         write(io, d)
    #     end
    # end
    
#    -- Momentum space, frequency space --
    # S = DSF(chi, Omega, 0, Tsteps, dt=dtau)  
    # # filename = @sprintf(
    # #     "./Heisenberg_data/S_N%i_k%.2f_T%.i_dt%.2f_betaInf_zz_test.csv", 
    # #     N_physics, k/pi, Tsteps*dtau, dtau
    # # )
    # filename = @sprintf(
    #     "./Heisenberg_data/S_N%i_k%.2f_T%.i_dt%.2f_beta%.2f_zz.csv", 
    #     N_physics, k/pi, Tsteps*dtau, dtau/tausweep, beta
    # )
    # open(filename, "w") do io 

    #     write(io, "RS,IS\n")

    #     for i in 1:length(S) 
    #         d = @sprintf("%.10f,%.10f\n", S[i].re, S[i].im)
    #         write(io, d)
    #     end
    # end

end
