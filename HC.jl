using ITensors
using ITensorMPS
using JLD2
using Printf

# Avoid oversubscription
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
    
    linkdim = 10
    cutoff = 1E-6
    dtau = 0.2
    tausweep = 5
    Tsteps = 10
    maxdim = 40
    
    psi_beta, sites = trivial_state(N*M*2*2) # |psi(beta=0)>
    H = MPO(H_aucillary(N, M, 1, 0.1, 0.1, 0.1), sites)

    beta = [0, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]
    d_beta = 0.01
    
    for i in 1:length(beta)-1

        b = beta[i+1]-beta[i]

        nsweeps = round(Int, b/d_beta)
        
        psi_beta = tdvp(
            H, -b/2, psi_beta; 
            nsweeps=nsweeps, maxdim=100, normalize=true, nsite=2, cutoff=cutoff
            , outputlevel=0
        )
        Sz = expect(psi_beta, "Sz"; sites=2)
        # println("Cooldown fin.")
        # println("Beta = ", beta)
        # println("psi1 Mz at 1: ", expect(psi_beta, "Sz"; sites=2))
        print(beta[i+1])
        println(",", Sz)
    end
end