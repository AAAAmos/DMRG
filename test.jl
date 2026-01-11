using ITensors, ITensorMPS
# using Printf
# using Statistics: mean
# using JLD2
# using Base.Threads
using LinearAlgebra: BLAS, eigvals, svd

function trivial_state(N)

    sites = siteinds("S=1/2", N)
    psi = MPS(Float64, sites, "Up")

    # construct trivial state at beta=0
    gates = ITensor[]
    for j in 1:2:N-1
        s1 = siteind(psi, j)
        s2 = siteind(psi, j+1)
        
        g = ITensor(s1, s2, s1', s2')
        # transition: |Up,Up> -> 1/√2(|Up,Up> + |Down,Down>)
        # Index 1 = Up, Index 2 = Down
        g[s1=>1, s2=>1, s1'=>1, s2'=>1] = 1.0 / sqrt(2)
        g[s1=>1, s2=>1, s1'=>2, s2'=>2] = 1.0 / sqrt(2)

        push!(gates, g)
    end

    psi = apply(gates, psi; cutoff=1e-10)
    psi = noprime(psi)

    return psi, sites
end

function purity(psi, sites)
    N = length(psi)
    p_inds = [sites[i] for i in 1:2:N]
    a_inds = [sites[i] for i in 2:2:N]

    T = ITensor(1.0)
    for i in 1:N
        T *= psi[i]
    end
    U, S, V = svd(T, p_inds)
    @show S
    # eigenvalues = [S[i,i]^2 for i in 1:dim(S, 1)]
    # purity = 0
    # for i in 1:dim(S, 1)
    #     purity += S[i, i]^2
    # end
    # return maximum(eigenvalues)
end

let
    N = 4*2

    psi_beta, sites = trivial_state(N) # |psi(beta=0)>
    println(purity(psi_beta, sites))
end

# purity = inner(rho, rho)
# println(purity)

# u = [1, 2, 3]
# v = [4, 5]

# # Compute the outer product
# outer_product_matrix = v * u'
# println(outer_product_matrix)

# states = [isodd(n) ? 2 : 1 for n in 1:10]
# println(states)
# println(log10(1e-5))
# println("number of threads: ", Threads.nthreads())
# function process(x)
#     sleep(3)
#     return 1 + x
# end

# T0 = time()

# t1 = Threads.@spawn process(1)
# t2 = Threads.@spawn process(2)

# a, b = (fetch(t1), fetch(t2))

# println(time()-T0)
# println(a, b)

# chi = load_object("./BLBQ_data/Chi_N7_obc_theta-0.10_lam10_k0.57_T100_dt0.05_zz_E0-13.121.jld2")
# chi = load_object("./Heisenberg_data/Chi_N16_k1.00_Tau100_dt0.05_beta0.01_zz.jld2")
# println(chi[10])

# for i in 0:1:10
#     println(i)
# end
# print(mean(1:20))
# function ep()
#     partial_sums = zeros(ComplexF64, nthreads())
#     print(partial_sums)
#     @threads for j in 1:10
#         partial_sums[threadid()] += 1
#         # println("Max RSS (Resident Set Size): ", Sys.maxrss() / (1024^2), " MiB")
#         print(Threads.threadid())
#     end
#     return partial_sums
# end

# task_p = @spawn ep()
# chi_p = zeros(ComplexF64, 10)
# chi_p .= fetch(task_p)

# # chi = load_object("./BLBQ_data/Chir_N10_theta0.08_lam20_n1_T50_dt0.05_pm_E0-10.833.jld2")
# chi = load_object("./BLBQ_data/Chi_N11_theta-0.10_lam20_k0.50_T50_dt0.05_zz_E0-17.734.jld2")

# println(chi)

# println(div(11, 2))
# a = 1+2im
# println(conj(a))

# data = [-2.263794620566305 - 2.531308496145357e-15im, 
#     3.0435498290081013 + 8.659739592076222e-16im]

# open("output.txt", "a") do io 
#     write(io, "k", )
#     for i in 1:length(data) 
#         d = @sprintf("%.10f\n", data[i].re)
#         write(io, d)
#     end
# end

# a = 2.0* pi 
# txt = @sprintf("2 in pi is %.1f", a/pi)
# println(txt)

# x = append!(a, b, c)
# println(x)

# function H_BLBQ(N, theta)
#     #=
#     Construct Hamiltonian of Bilinear Biquadratic spin-1 model
#     return:
#         os: operaters
#     =#

#     os = OpSum()

#     for n = 1:(N-1)
#         os .+= cos(theta), "Sz", n, "Sz", n+1
#         os .+= cos(theta)/2, "S+", n, "S-", n+1
#         os .+= cos(theta)/2, "S-", n, "S+", n+1

#         os .+= sin(theta), "Sz", n, "Sz", n+1, "Sz", n, "Sz", n+1
#         os .+= sin(theta)/2, "S+", n, "S-", n+1, "S+", n, "S-", n+1
#         os .+= sin(theta)/2, "S-", n, "S+", n+1, "S-", n, "S+", n+1
#     end

#     return os
# end
# function TrotterGates(sites, theta, tau)
#     # Return Trotter gates exp(-it * h_ij)

#     gates = ITensor[]
#     for n = 1:(length(sites)-1)

#         s1 = sites[n]
#         s2 = sites[n+1]

#         h_bil = 
#             op("Sz", s1) * op("Sz", s2) + 
#             1/2 * op("S+", s1) * op("S-", s2) + 
#             1/2 * op("S-", s1) * op("S+", s2)
#         h_biq = prime(h_bil) * h_bil
#         h_biq = mapprime(h_biq, 2, 1)

#         hn = cos(theta)*h_bil + sin(theta)*h_biq

#         Gn = exp(-im * tau/2 * hn)
#         push!(gates, Gn)
#     end

#     append!(gates, reverse(gates))

#     return gates
# end 

# sites = siteinds("S=1", 5)
# center = 3
# nsweeps = 3
# cutoff = 1E-9
# maxdim = [20, 50, 50]
# psi_ran = random_mps(sites; linkdims=10)
# H = MPO(H_BLBQ(5, -0.1*pi), sites)
# Tgatep = TrotterGates(sites, -0.1*pi, 0.05)
# Tgaten = TrotterGates(sites, -0.1*pi, -0.05)
# energy, psi0 = dmrg(H, psi_ran; nsweeps, maxdim, cutoff)

# psi0[center] = noprime(op("Sz", sites[center]) * psi0[center])
# psi_t_p = apply(Tgatep, psi0; cutoff)
# psi_t_n = apply(Tgaten, psi0; cutoff)
# j = 1
# psi_t_p[j] = noprime(op("Sz", sites[j]) * psi_t_p[j])
# psi_t_n[j] = noprime(op("Sz", sites[j]) * psi_t_n[j])

# c1 = inner(psi0, psi_t_p+psi_t_n)
# c2 = inner(conj(psi0), psi_t_p+psi_t_n)
# println(c1)
# println(c2)

# chi = zeros(ComplexF64, 10)

# A = Array{ComplexF64}(undef, 5)
# for i in 1:5
#     println(3-i)
# end 
# @show A 

# function A(x)

#     x = 1
#     x += 1

#     function square(x)
#         return x^2
#     end 

#     return square(x)
# end

# println(A(0))

# sites = siteinds("S=1", 5)
# psi0 = random_mps(sites; linkdims=10)
# apply(op("S-", sites), psi0)
# @show psi0[5]
# @show inds(sites)
# println(div(length(psi0), 2))
# psi0[5] = noprime(op("S-", sites[5]) * psi0[5])
# @show psi0[5]


# function TrotterGates(sites, theta, tau)

#     gates = ITensor[]
#     for n = 1:(length(sites)-1)

#         s1 = sites[n]
#         s2 = sites[n+1]

#         h_bil = 
#             op("Sz", s1) * op("Sz", s2) + 
#             1/2 * op("S+", s1) * op("S-", s2) + 
#             1/2 * op("S-", s1) * op("S+", s2)
#         h_biq = prime(h_bil) * h_bil
#         h_biq = mapprime(h_biq, 2, 1)

#         hn = cos(theta)*h_bil + sin(theta)*h_biq

#         Gn = exp(-im * tau/2 * hn)
#         push!(gates, Gn)
#     end
#     append!(gates, reverse(gates))

#     return gates
# end 

# let 
#     # parameters
#     N = 32
#     theta = 1.0*pi 

#     linkdim = 10
#     nsweeps = 5
#     maxdim = [100,150,200,200,250]
#     cutoff = 1E-10
#     dtau = 0.1
#     ttotal = 1.0

#     sites = siteinds("S=1", N)
    
#     psi = random_mps(sites; linkdims=linkdim)

#     # time evolution
#     gates = TrotterGates(sites, theta, dtau)
#     # for t in 0.0:dtau:ttotal
#     #     Sz = expect(psi, "Sz"; sites=16)
#     #     println("$t $Sz")

#     #     t≈ttotal && break

#     #     psi = apply(gates, psi; cutoff)
#     #     normalize!(psi)
#     # end

# end


# let 
#     linkdim = 10
#     nsweeps = 5
#     maxdim = [100,100,100,100,200]
#     cutoff = [1E-10]
#     @printf("Psi_0 link dims = %.1f\n", linkdim/2)
#     # @printf("max")

# end

# i, j, = 5, 4
# @show mod(j, i)+1

# print(isequal(1., 1.))
# print(isequal(NaN, NaN))
# a = NaN
# b = NaN
# print(typeof(a))
# print(a == b)

# Print the numbers 1 through 5

# i = 0

# let i = i
#     while i < 5
#         i += 1 
#         println(i)
#     end
# end

# print(i)