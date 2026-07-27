using ITensors
using ITensorMPS
using Printf: @sprintf
using JLD2
include("DSF_hc.jl")

# ==============================================================================
#  XTRG  (exponential tensor renormalization group)
#
#  Reference: Eqs. (4)-(5) — ρ(τ) = e^{-τH} is represented directly as an MPO
#  on the PHYSICAL sites. There is no ancilla / purification here, unlike the
#  TDVP approach this file used to implement.
#
#  Algorithm
#  ---------
#   1. Build ρ(τ0) at a tiny τ0 via a Taylor series (cheap & very accurate,
#      because τ0·‖H‖ ≪ 1, so only a handful of terms are needed).
#   2. Repeatedly SQUARE it:  ρ(2τ0)=ρ(τ0)·ρ(τ0),  ρ(4τ0)=ρ(2τ0)·ρ(2τ0), ...
#      (Eqs. 4-5). Each squaring is an EXACT operator identity — the only
#      approximations anywhere are the initial Taylor truncation and the MPO
#      bond-dimension truncation applied after every multiplication. There is
#      no Trotter/time-integration error at all, unlike TDVP.
#   3. To land on an arbitrary β (not only powers of two times τ0), keep the
#      whole "doubling ladder" ρ(τ0), ρ(2τ0), ρ(4τ0), ... and combine the
#      rungs needed via binary exponentiation ("square-and-multiply"): write
#      m = β/τ0 in binary and multiply together the ladder elements whose bit
#      is set. This reaches ANY integer multiple of τ0 in O(log2(β/τ0))
#      multiplications, reusing one ladder for the whole beta_list — e.g.
#      ~13-15 multiplications per β instead of β/dβ ~ hundreds of TDVP steps.
#
#  IMPORTANT — please read before trusting this on production runs
#  ------------------------------------------------------------------
#  I don't have DSF_hc.jl, so two integration points below are ASSUMPTIONS
#  (clearly flagged inline):
#    (a) trivial_state(SpinHalf(), N*M*2, QN=...) can be called with the bare
#        physical site count (no ancilla doubling) and its `sites` output can
#        be reused directly.
#    (b) H_HC(...; auc=false, ...) returns the physical-only Hamiltonian
#        OpSum on those same N*M*2 sites.
#  If either is wrong, the Hamiltonian/site wiring — not the XTRG math —
#  is what needs fixing. Validate against xtrg_selftest.jl first, and ideally
#  cross-check E(β)/Mz(β) against a few points from your old TDVP output
#  before trusting new production data.
# ==============================================================================

"Build ρ0 = e^{-τH} as an MPO via a truncated Taylor series:
 ρ0 = Σ_{k=0}^{order} (-τ)^k/k! H^k, computed by the recursion
 term_k = (-τ/k)·H·term_{k-1}, term_0 = Id."
function taylor_rho0(H::MPO, τ::Float64, sites;
                      order::Int=16, cutoff::Float64=1e-13, maxdim::Int=800)
    Id   = MPO(sites, "Id")
    ρ    = copy(Id)
    term = copy(Id)
    for k in 1:order
        term = apply(H, term; cutoff=cutoff, maxdim=maxdim)
        term = (-τ / k) * term
        ρ    = +(ρ, term; cutoff=cutoff, maxdim=maxdim)
        nrm = norm(term)
        println("    Taylor order $k:  |term| = $(round(nrm, sigdigits=4)),  chi(rho) = $(maxlinkdim(ρ))")
        if k > 3 && nrm < cutoff
            break
        end
    end
    return ρ
end

"Tr(A) for an MPO A — contracts the bra/ket physical indices directly
 (no ancilla needed)."
function mpo_trace(A::MPO, sites)
    L = ITensor(1.0)
    for j in 1:length(A)
        L *= A[j] * delta(sites[j], dag(sites[j])')
    end
    return L[]
end

"<O> = Tr(rho*O)/Tr(rho)."
function mpo_expect(ρ::MPO, O::MPO, sites; cutoff::Float64=1e-12, maxdim::Int=2000)
    ρO = apply(ρ, O; cutoff=cutoff, maxdim=maxdim)
    return mpo_trace(ρO, sites) / mpo_trace(ρ, sites)
end

"""
Build the doubling ladder rho_pow[k+1] = rho(2^k * tau0), k = 0..K, by
Taylor-expanding the k=0 rung and repeatedly squaring (Eqs. 4-5). Each rung
is renormalized by its trace purely for numerical stability — this does not
affect any expectation value since mpo_expect always takes a trace ratio.
"""
function build_doubling_ladder(H::MPO, τ0::Float64, sites, K::Int;
                                taylor_order::Int=16,
                                cutoff::Float64=1e-13, maxdim::Int=800)
    ρ_pow = Vector{MPO}(undef, K+1)
    println("Building rho(tau0), tau0 = $τ0 ...")
    ρ_pow[1] = taylor_rho0(H, τ0, sites; order=taylor_order, cutoff=cutoff, maxdim=maxdim)
    Z0 = real(mpo_trace(ρ_pow[1], sites))
    ρ_pow[1] = ρ_pow[1] / Z0
    for k in 1:K
        τ_from = τ0*2.0^(k-1); τ_to = τ0*2.0^k
        println("Squaring:  tau=$(round(τ_from,sigdigits=4)) -> tau=$(round(τ_to,sigdigits=4))")
        ρsq = apply(ρ_pow[k], ρ_pow[k]; cutoff=cutoff, maxdim=maxdim)
        Z = real(mpo_trace(ρsq, sites))
        ρ_pow[k+1] = ρsq / Z
        println("    chi = $(maxlinkdim(ρ_pow[k+1]))")
    end
    return ρ_pow
end

"Combine ladder rungs via binary exponentiation to build rho(m*tau0)."
function rho_from_ladder(ρ_pow::Vector{MPO}, m::Int, sites;
                          cutoff::Float64=1e-12, maxdim::Int=800)
    @assert m >= 1
    ρ = nothing
    mm, k = m, 0
    while mm > 0
        if isodd(mm)
            ρ = (ρ === nothing) ? ρ_pow[k+1] : apply(ρ, ρ_pow[k+1]; cutoff=cutoff, maxdim=maxdim)
        end
        mm >>= 1
        k += 1
    end
    Z = real(mpo_trace(ρ, sites))
    return ρ / Z
end


let
    Total_time = time()

#  -- Physical parameter setup ---
    N = 2
    M = 2
    Jnn, Jnnn, DMI, h = 1, 0.1, 0., 0.1
    ani = 0.

    obc_x = true
    obc_y = true
    Ox, Oy = "t", "t"

#  -- numerical setup ---
    QN_conservation = true

#  -- XTRG accuracy / cost knobs --
    beta_unit    = 0.01      # smallest spacing present in beta_list -- keep in sync if you edit beta_list!
    p_extra      = 1         # extra binary doublings below beta_unit -> tau0 = beta_unit/2^p_extra
    τ0           = beta_unit / 2.0^p_extra
    taylor_order = 16
    rho_cutoff   = 1E-10
    rho_maxdim   = 400       # MPO bond dim for rho -- a DIFFERENT quantity from the old MPS "maxdim=8". Tune/benchmark this!
    println("tau0 = ", τ0)
    
#  -- Temperature --
    beta_list = [0.02, 0.04, 0.08, 0.16]
    save_list = []

#  -- files --
    E_file = @sprintf(
        "../HC_data/%.i%.i_S0.5_nnn%.2f_DM%.2f_Ox%s_Oy%s_psi%.i_xtrg_k%.i.csv",
        N, M, Jnnn, DMI, Ox, Oy, Int(log10(rho_cutoff)), rho_maxdim
    )

# --- finite T (XTRG) ---

#  -- physical sites (NO ancilla needed for XTRG) --
    sites = siteinds("S=1/2", N*M*2; conserve_qns=QN_conservation)

    Hos, r = H_HC(N, M, Jnn, Jnnn, DMI, h, ani; anc=false, obc_x, obc_y)
    H = MPO(Hos, sites)
    M1OP = MPO(M1_op(N*M*2; anc=false), sites)
    M2OP = MPO(M2_op(N*M*2; anc=false), sites)

#  -- build the doubling ladder ONCE, reuse for every beta in beta_list --
    beta_max = maximum(beta_list)
    m_max = round(Int, beta_max/τ0)
    K = ceil(Int, log2(m_max))
    ladder = build_doubling_ladder(H, τ0, sites, K;
                 taylor_order=taylor_order, cutoff=rho_cutoff, maxdim=rho_maxdim)

    for beta in beta_list

        m = round(Int, beta/τ0)
        if !isapprox(m*τ0, beta; atol=1e-8)
            error("beta = $beta is not (to numerical precision) an integer multiple of tau0 = $τ0 -- adjust beta_unit/p_extra")
        end

        rho_beta = rho_from_ladder(ladder, m, sites; cutoff=rho_cutoff, maxdim=rho_maxdim)

        println("Process peak RSS (MB): ", Sys.maxrss()/1.04E6)
        println("Cooldown fin.")
        println("Beta = ", beta)

#   - Save rho -
        if beta in save_list
            Psi_file = @sprintf(
                "./HC_data/Psi/%.i%.i_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_rho_xtrg.jld2",
                N, M, DMI, Ox, Oy, beta, rho_maxdim
            )
            save_object(Psi_file, rho_beta)
            Sites_file = @sprintf(
                "./HC_data/Psi/%.i%.i_DM%.2f_Ox%s_Oy%s_b%.2f_k%.i_sites_xtrg.jld2",
                N, M, DMI, Ox, Oy, beta, rho_maxdim
            )
            save_object(Sites_file, sites)
        end

#   - Mz, Purity, E -
        Mz = mpo_expect(rho_beta, M1OP, sites)
        M2 = mpo_expect(rho_beta, M2OP, sites)
        E_beta = mpo_expect(rho_beta, H, sites; cutoff=rho_cutoff, maxdim=rho_maxdim)

        dim = maxlinkdim(rho_beta)

        open(E_file, "a") do io
            if obc_y
                Sz_i = [mpo_expect(rho_beta, MPO(let os=OpSum(); os+="Sz",j; os end, sites), sites) for j in 1:M]

                d = @sprintf("%.4f,%.10f,%.10f,%.10f,%.i", beta, real(E_beta), real(Mz), real(M2), dim)
                write(io, d)
                for j in 1:M
                    d = @sprintf(",%.6f", real(Sz_i[j]))
                    write(io, d)
                end
                write(io, "\n")
            else
                d = @sprintf("%.4f,%.10f,%.10f,%.10f,%.i\n", beta, real(E_beta), real(Mz), real(M2), dim)
                write(io, d)
            end
        end

        GC.gc()
    end

    println("Total computational time: $(time()-Total_time)")
end
