using ITensors
using ITensorMPS
using JLD2
using Printf
using Statistics: mean
# using LinearAlgebra
# BLAS.set_num_threads(1)

function res(x, M) # same as % in python
    return mod(x-1, M)+1
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

function H_phy(n, m, J, j, D, h)

    col = m*2
    N = n*col 

    os = OpSum()

    for a in 0:n-1 
        for b in 1:m 

            y = 2*b - 1
            i = col*a + y

            # Zeeman
            os .+= -h, "Sz", i 
            os .+= -h, "Sz", i+1

            # A site NN
            os .+= -J/2, "Sz", i, "Sz", i+1
            os .+= -J/4, "S+", i, "S-", i+1
            os .+= -J/4, "S-", i, "S+", i+1

            os .+= -J/2, "Sz", i, "Sz", (col*a+res(y-1, col))
            os .+= -J/4, "S+", i, "S-", (col*a+res(y-1, col))
            os .+= -J/4, "S-", i, "S+", (col*a+res(y-1, col))

            os .+= -J/2, "Sz", i, "Sz", res(i+1-col, N)
            os .+= -J/4, "S+", i, "S-", res(i+1-col, N)
            os .+= -J/4, "S-", i, "S+", res(i+1-col, N)

            # NNN
            os .+= -j, "Sz", i, "Sz", (col*a+res(y+2, col))
            os .+= -j/2 + im*D, "S+", i, "S-", (col*a+res(y+2, col))
            os .+= -j/2 - im*D, "S-", i, "S+", (col*a+res(y+2, col))
            
            os .+= -j, "Sz", i, "Sz", res(i-col, N)
            os .+= -j/2 + im*D, "S+", i, "S-", res(i-col, N)
            os .+= -j/2 - im*D, "S-", i, "S+", res(i-col, N)

            os .+= -j, "Sz", i, "Sz", res(col*(a+1)+res(y-2, col), N)
            os .+= -j/2 + im*D, "S+", i, "S-", res(col*(a+1)+res(y-2, col), N)
            os .+= -j/2 - im*D, "S-", i, "S+", res(col*(a+1)+res(y-2, col), N)

            y = 2*b
            i = col*a + y

            # B site NN
            os .+= -J/2, "Sz", i, "Sz", i-1
            os .+= -J/4, "S+", i, "S-", i-1
            os .+= -J/4, "S-", i, "S+", i-1

            os .+= -J/2, "Sz", i, "Sz", (col*a+res(y+1, col))
            os .+= -J/4, "S+", i, "S-", (col*a+res(y+1, col))
            os .+= -J/4, "S-", i, "S+", (col*a+res(y+1, col))

            os .+= -J/2, "Sz", i, "Sz", res(i-1+col, N)
            os .+= -J/4, "S+", i, "S-", res(i-1+col, N) 
            os .+= -J/4, "S-", i, "S+", res(i-1+col, N)
            
            # NNN
            os .+= -j, "Sz", i, "Sz", (col*a+res(y-2, col))
            os .+= -j/2 + im*D, "S+", i, "S-", (col*a+res(y-2, col))
            os .+= -j/2 - im*D, "S-", i, "S+", (col*a+res(y-2, col))
            
            os .+= -j, "Sz", i, "Sz", res(i+col, N)
            os .+= -j/2 + im*D, "S+", i, "S-", res(i+col, N)
            os .+= -j/2 - im*D, "S-", i, "S+", res(i+col, N)

            os .+= -j, "Sz", i, "Sz", res(col*(a-1)+res(y+2, col), N)
            os .+= -j/2 + im*D, "S+", i, "S-", res(col*(a-1)+res(y+2, col), N)
            os .+= -j/2 - im*D, "S-", i, "S+", res(col*(a-1)+res(y+2, col), N)

        end
    end
    
    return os
end

function H_aucillary(n, m, J, j, D, h)

    # m -> 2*m

    col = m*2
    N = n*col 

    os = OpSum()

    for a in 0:n-1 
        for b in 1:m 

            y = 2*b - 1
            i = col*a + y
            ii = 2*i

            # Zeeman
            os .+= -h, "Sz", ii 
            os .+= -h, "Sz", ii+1*2

            # A site NN
            os .+= -J/2, "Sz", ii, "Sz", ii+1*2
            os .+= -J/4, "S+", ii, "S-", ii+1*2
            os .+= -J/4, "S-", ii, "S+", ii+1*2

            os .+= -J/2, "Sz", ii, "Sz", (col*a+res(y-1, col))*2
            os .+= -J/4, "S+", ii, "S-", (col*a+res(y-1, col))*2
            os .+= -J/4, "S-", ii, "S+", (col*a+res(y-1, col))*2

            os .+= -J/2, "Sz", ii, "Sz", res(i+1-col, N)*2
            os .+= -J/4, "S+", ii, "S-", res(i+1-col, N)*2
            os .+= -J/4, "S-", ii, "S+", res(i+1-col, N)*2

            # NNN
            os .+= -j, "Sz", ii, "Sz", (col*a+res(y+2, col))*2
            os .+= -j/2 + im*D, "S+", ii, "S-", (col*a+res(y+2, col))*2
            os .+= -j/2 - im*D, "S-", ii, "S+", (col*a+res(y+2, col))*2
            
            os .+= -j, "Sz", ii, "Sz", res(i-col, N)*2
            os .+= -j/2 + im*D, "S+", ii, "S-", res(i-col, N)*2
            os .+= -j/2 - im*D, "S-", ii, "S+", res(i-col, N)*2

            os .+= -j, "Sz", ii, "Sz", res(col*(a+1)+res(y-2, col), N)*2
            os .+= -j/2 + im*D, "S+", ii, "S-", res(col*(a+1)+res(y-2, col), N)*2
            os .+= -j/2 - im*D, "S-", ii, "S+", res(col*(a+1)+res(y-2, col), N)*2

            y = 2*b
            i = col*a + y
            ii = 2*i

            # B site NN
            os .+= -J/2, "Sz", ii, "Sz", ii-1*2
            os .+= -J/4, "S+", ii, "S-", ii-1*2
            os .+= -J/4, "S-", ii, "S+", ii-1*2

            os .+= -J/2, "Sz", ii, "Sz", (col*a+res(y+1, col))*2
            os .+= -J/4, "S+", ii, "S-", (col*a+res(y+1, col))*2
            os .+= -J/4, "S-", ii, "S+", (col*a+res(y+1, col))*2

            os .+= -J/2, "Sz", ii, "Sz", res(i-1+col, N)*2
            os .+= -J/4, "S+", ii, "S-", res(i-1+col, N)*2 
            os .+= -J/4, "S-", ii, "S+", res(i-1+col, N)*2
            
            # NNN
            os .+= -j, "Sz", ii, "Sz", (col*a+res(y-2, col))*2
            os .+= -j/2 + im*D, "S+", ii, "S-", (col*a+res(y-2, col))*2
            os .+= -j/2 - im*D, "S-", ii, "S+", (col*a+res(y-2, col))*2
            
            os .+= -j, "Sz", ii, "Sz", res(i+col, N)*2
            os .+= -j/2 + im*D, "S+", ii, "S-", res(i+col, N)*2
            os .+= -j/2 - im*D, "S-", ii, "S+", res(i+col, N)*2

            os .+= -j, "Sz", ii, "Sz", res(col*(a-1)+res(y+2, col), N)*2
            os .+= -j/2 + im*D, "S+", ii, "S-", res(col*(a-1)+res(y+2, col), N)*2
            os .+= -j/2 - im*D, "S-", ii, "S+", res(col*(a-1)+res(y+2, col), N)*2

        end
    end
    
    return os
end

# H_aucillary(3, 3, 1, 0.1, 0.1, 0.1)

let 
    N = 2
    M = 2
    Jnn, Jnnn, DMI, h = 1, 0.1, 0.1, 0.1
    
    dmrg_linkdim = 10
    dmrg_maxdim = [200, 200, 200, 200, 200]
    maxdim = 1000

    psi_cutoff = 1E-6
    time_cutoff = 1E-10
    println("State cutoff: ", psi_cutoff)

    dtau = 0.2
    tausweep = 5
    Tsteps = 10

    beta_list = [0, 1/100, 1/10, 5/10, 1, 2, 3, 4, 5, 6]
    d_beta = 0.001
    
#    --- T=0 ---

    # -- physical states --

    sites = siteinds("S=1/2", N*M*2; conserve_sz = false)
    psi_ran = random_mps(sites; linkdims=dmrg_linkdim)

    H = MPO(H_phy(N, M, Jnn, Jnnn, DMI, h), sites)

#    -- dmrg --
    E0, psi0 = dmrg(H, psi_ran; 
        nsweeps=10, maxdim=dmrg_maxdim, cutoff=psi_cutoff, outputlevel=1
    )
    mean_sz = mean(expect(psi0, "Sz"))
    # println("Sz(5)=", expect(psi0, "Sz")[5])
    println("Ground state energy: ", E0)
    E_file = "./HC_data/22_E_cut-6.csv"
    open(E_file, "w") do io 
        write(io, "beta,E,Sz\n")
        d = @sprintf("%.i,%.10f,%.5f\n", 100000, E0, mean_sz)
        write(io, d)
    end

#    -- correlation function --
    # Si = 9
    # chi = chi_x_t_FT(Si, Si, H, psi0, sites, Tsteps, dtau;
    #     nsites=2, cutoff=time_cutoff, maxdim=1000, ns=tausweep
    # )

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
    
    psi_beta, sites = trivial_state(N*M*2*2) # |psi(beta=0)>
    H = MPO(H_aucillary(N, M, Jnn, Jnnn, DMI, h), sites)

    for i in 1:length(beta_list)-1

        b = beta_list[i+1]-beta_list[i]
        nsweeps = round(Int, b/d_beta)

        psi_beta = tdvp(
            H, -b/2, psi_beta; 
            nsweeps=nsweeps, maxdim=1000, normalize=true, nsite=2, cutoff=psi_cutoff
            , outputlevel=1
        )
        println("Cooldown fin.")
        println("Beta = ", beta_list[i+1])
    # end

#    - Mz, Purity, E -
        # Mz = expect(psi_beta, "Sz"; sites=1)
        mean_sz = mean(expect(psi_beta, "Sz"))

        # site number < 14 is needed !!!
        # println("Purity: ", purity(psi_beta, sites))

        psi_h = apply(H, psi_beta; cutoff=psi_cutoff)
        E_beta = inner(psi_beta, psi_h)
        println("Energy: ", E_beta)

        open(E_file, "a") do io 
            d = @sprintf("%.4f,%.10f,%.5f\n", beta_list[i+1], E_beta, mean_sz)
            write(io, d)
        end
    end
end