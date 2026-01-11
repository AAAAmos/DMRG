using ITensors
using ITensorMPS
using Printf: @sprintf
using LinearAlgebra: BLAS, eigvals, svd
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
    os .+= J1, "Sz", N-1, "Sz", 1
    os .+= J1/2, "S+", N-1, "S-", 1
    os .+= J1/2, "S-", N-1, "S+", 1

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

function trivial_state(N)

    # sites = siteinds("S=1/2", N; conserve_sz = true, qnname_sz="TotalSz")
    sites = siteinds("S=1/2", N)

    # Initialize MPS in a simple product state |Up, Up, Up... >
    # states = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    # psi = MPS(Float64, sites, states)
    psi = MPS(Float64, sites, "Up")

    # construct trivial state at beta=0
    # map |Up, Up> to 1/√2(|Up, Up> + |Down, Down>)
    gates = ITensor[]
    for j in 1:2:N-1
        s1 = siteind(psi, j)
        s2 = siteind(psi, j+1)
        
        # Create an operator tensor with 4 indices: (s1, s2) and (s1', s2')
        g = ITensor(s1, s2, s1', s2')
        # transition: |Up,Up> -> 1/√2(|Up,Up> + |Down,Down>)
        # Index 1 = Up, Index 2 = Down
        g[s1=>1, s2=>1, s1'=>1, s2'=>1] = 1.0 / sqrt(2)
        g[s1=>1, s2=>1, s1'=>2, s2'=>2] = 1.0 / sqrt(2)

        push!(gates, g)
    end
    # @show gates

    psi = apply(gates, psi; cutoff=1e-10)
    psi = noprime(psi)

    # Verification: Measure <Sz> on a physical site. Expect 0
    # sz_1 = expect(psi, "Sz"; sites=2)
    # println("Verification - <Sz> at site 1: ", round(sz_1, digits=5))
    # println("Norm of |psi_0>: ", norm(psi))
    println("EPR state fin.")

    return psi, sites
end

function purity(psi, sites)
    #= Return the purity of the physical states TrA(rho^2)
    psi: aucillary state
    sites:the sites of the states
    =#

    N = length(psi)
    p_inds = [sites[i] for i in 1:2:N]
    a_inds = [sites[i] for i in 2:2:N]

    T = ITensor(1.0)
    for i in 1:N
        T *= psi[i]
    end
    U, S, V = svd(T, p_inds)

    purity = 0
    for i in 1:dim(S, 1)
        rho = S[i, i]^2
        purity += rho^2
    end

    return purity
end


let
    N_physics = 7
    N = 2 * N_physics
    
    linkdim = 20
    maxdim = [200, 200, 200, 200, 200]
    cutoff = 1E-12
    dtau = 0.1
    tausweep = 2
    Tsteps = 100

    theta = 0.102*pi # AKLT phase

    beta_list = [0, 0.01]
    # beta_list = [0, 5/10, 1, 2, 3, 4, 5]
    d_beta = 0.001
    # nsweeps = round(Int, beta/d_beta)
    
    k = pi
    Omega = range(0.0, 5.0, length = 500)
    
#    --- T=0 ---

    # # -- physical states --
    # sites = siteinds("S=1/2", N_physics; conserve_sz = true)
    # states = [isodd(n) ? "Up" : "Dn" for n in 1:N_physics]
    # # states = ["Dn" for n in 1:N_physics]
    # # states = ["Up" for n in 1:N_physics]
    # psi = MPS(Float64, sites, states)

    # sites = siteinds("S=1/2", N; conserve_sz = false)
    # psi_ran = random_mps(sites; linkdims=linkdim)

    # H = MPO(H_heisenberg(N_physics), sites)

    # # -- dmrg --
    # E0, psi0 = dmrg(H, psi_ran; nsweeps=10, maxdim, cutoff, outputlevel=1)
    # # println("Sz(5)=", expect(psi0, "Sz")[5])
    # println("E0 at T=0: ", E0)

    # # -- correlation function --
    # Si = 9
    # chi = chi_x_t_FT(Si, Si, H, psi0, sites, Tsteps, dtau;
    #     nsites=2, cutoff=cutoff, maxdim=1000, ns=tausweep
    # )

    # # Tgatep = TrotterGates_Hei(sites, dtau)
    # # Tgaten = TrotterGates_Hei(sites, -dtau)
    # # chi = chi_x_t(Si, Si, psi0, E0, sites, [Tgatep, Tgaten]; dt=dtau, Tsteps=Tsteps, cutoff=1E-10)

    # filename = @sprintf(
    #     "./Heisenberg_data/Chi_N%i_S%iS%i_Tau%i_dt%.2f_betaInf_zz_auc_phy.csv", 
    #     # "./test.csv", 
    #     N_physics, Si, Si, Tsteps*dtau, dtau/tausweep
    # )
    # open(filename, "w") do io 

    #     write(io, "t,RS,IS\n")

    #     for t in -Tsteps:Tsteps
    #         i = t + Tsteps+1
    #         d = @sprintf("%.2f,%.10f,%.10f\n", t*dtau, chi[i].re, chi[i].im)
    #         write(io, d)
    #     end
    # end

#    --- finite T ---
    psi_beta, sites = trivial_state(N) # |psi(beta=0)>
    H = MPO(H_heisenberg_aucillary(N), sites)
    # H = MPO(H_BLBQ_aucillary(N, theta), sites)
    
    # println("Purity: ", purity(psi_beta, sites))

    for i in 1:length(beta_list)-1

        b = beta_list[i+1]-beta_list[i]
        nsweeps = round(Int, b/d_beta)

        psi_beta = tdvp(
            H, -b/2, psi_beta; 
            nsweeps=nsweeps, maxdim=1000, normalize=true, nsite=2, cutoff=cutoff
            , outputlevel=1
        )
        println("Cooldown fin.")
        println("Beta = ", beta_list[i+1])
    end

#    - Mz, Trace -
    #     # Mz = expect(psi_beta, "Sz"; sites=1)
    #     println("Purity: ", purity(psi_beta, sites))
        
    # end

#    -- Real space, time --
    BLAS.set_num_threads(1)
    Si = 1
    chi = chi_x_t_FT(Si, Si, H, psi_beta, sites, Tsteps, dtau;
        nsites=2, cutoff=cutoff, maxdim=1000, ns=tausweep
    )

    filename = @sprintf(
        "./Heisenberg_data/Chi_N%i_pbc_S%iS%i_Tau%i_dt%.2f_beta%.2f_zz.csv", 
        N_physics, Si-4, Si-4, Tsteps*dtau, dtau/tausweep, beta_list[2]
    )
    open(filename, "w") do io 

        write(io, "t,RS,IS\n")

        for t in -Tsteps:Tsteps
            i = t + Tsteps+1
            d = @sprintf("%.2f,%.10f,%.10f\n", t*dtau, chi[i].re, chi[i].im)
            write(io, d)
        end
    end

#    -- Momentum space, real time --
    # BLAS.set_num_threads(1)
    # chi = chi_t_FT(k, H, psi_beta, sites, Tsteps, dtau; 
    #     nsites=2, cutoff=cutoff, maxdim=1000, ns=tausweep
    # )
    # filename = @sprintf(
    #     "./Heisenberg_data/Chi_N%i_k%.2f_Tau%.i_dt%.2f_beta%.2f_zz_err-6.csv", 
    #     N_physics, k/pi, Tsteps*dtau, dtau/tausweep, beta
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
