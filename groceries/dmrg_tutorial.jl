using ITensors, ITensorMPS
using Printf

function idx(n, x) 
    return mod(n, x) + 1
end

function Hamiltonian(sites, m, n, Jnn, Jnnn, D, h)

    col = m*2
    N = col*n

    d = D*im

    os = OpSum()

    for x = 0:n-1 
        for y = 0:n-1

            i = col*x + 2*y + 1
            # Zeeman terms
            os .+= (-h, "Sz", i)
            os .+= (-h, "Sz", i+1)

            # NN terms
                # tr
            j = i+1
            os .+= (-Jnn, "Sz", i, "Sz", j)
            os .+= (-Jnn/2, "S+", i, "S-", j)
            os .+= (-Jnn/2, "S-", i, "S+", j)
                # b
            j = col*x + idx(2*y-1, col)
            os .+= (-Jnn, "Sz", i, "Sz", j)
            os .+= (-Jnn/2, "S+", i, "S-", j)
            os .+= (-Jnn/2, "S-", i, "S+", j)
                # tl
            j = idx(col*x + 2*y+1, N)
            os .+= (-Jnn, "Sz", i, "Sz", j)
            os .+= (-Jnn/2, "S+", i, "S-", j)
            os .+= (-Jnn/2, "S-", i, "S+", j)

            # NNN of A
                # r
            j = idx(col*(x+1) + 2*y, N)
            os .+= (-Jnnn, "Sz", i, "Sz", j)
            os .+= (-Jnnn/2 - d, "S+", i, "S-", j)
            os .+= (-Jnnn/2 + d, "S-", i, "S+", j)
                # bl
            j = col*x + idx(2*y-2, col)
            os .+= (-Jnnn, "Sz", i, "Sz", j)
            os .+= (-Jnnn/2 - d, "S+", i, "S-", j)
            os .+= (-Jnnn/2 + d, "S-", i, "S+", j)
                # tl
            j = idx(col*(x-1) + idx(2*y+2, col), N)
            os .+= (-Jnnn, "Sz", i, "Sz", j)
            os .+= (-Jnnn/2 - d, "S+", i, "S-", j)
            os .+= (-Jnnn/2 + d, "S-", i, "S+", j)

            # NNN of B
            i = col*x + 2*y+1
                # l
            j = idx(col*(x-1) + 2*y+1, N)
            os .+= (-Jnnn, "Sz", i, "Sz", j)
            os .+= (-Jnnn/2 - d, "S+", i, "S-", j)
            os .+= (-Jnnn/2 + d, "S-", i, "S+", j)
                # tr
            j = col*x + idx(2*y+3, col)
            os .+= (-Jnnn, "Sz", i, "Sz", j)
            os .+= (-Jnnn/2 - d, "S+", i, "S-", j)
            os .+= (-Jnnn/2 + d, "S-", i, "S+", j)
                # br
            j = idx(col*(x+1) + idx(2*y-1, col), N)
            os .+= (-Jnnn, "Sz", i, "Sz", j)
            os .+= (-Jnnn/2 - d, "S+", i, "S-", j)
            os .+= (-Jnnn/2 + d, "S-", i, "S+", j)

        end 
    end 

    return MPO(os, sites)

end 


let
    m = 5
    n = 5
    N = m*n*2
    JNN = 1.0
    JNNN = 0.1
    D = 0.1
    h = 0.1

    sites = siteinds("S=1/2", N)

    H = Hamiltonian(sites, m, n, JNN, JNNN, D, h)

    psi0 = random_mps(sites; linkdims=10)

    nsweeps = 5
    maxdim = [100,100,100,100,200]
    cutoff = [1E-10]

    energy,psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    @printf("Final energy = %.12f\n", energy)

    # return
end