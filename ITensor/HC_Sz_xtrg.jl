using ITensors
using ITensorMPS
using Printf: @sprintf
using JLD2
include("DSF_hc.jl")

# ==============================================================================
#  XTRG  (exponential tensor renormalization group)
#
#  Reference: Eqs. (4)-(5) -- rho(tau) = e^{-tau*H} is represented directly as
#  an MPO on the PHYSICAL sites (no ancilla / purification, unlike the old
#  TDVP approach this file used to implement).
#
#  This version implements the algorithm literally as written in the paper,
#  with no generalization: pick tau0 = beta_unit, build rho(tau0) via a
#  Taylor series, then square m times:
#       rho_0 = rho(tau0)
#       rho_n = rho_{n-1} . rho_{n-1} = rho(2^n * tau0)          (Eqs. 4-5)
#  reaching a single final beta = beta_unit * 2^m in this run. Each squaring
#  is an EXACT operator identity -- the only approximations anywhere are the
#  initial Taylor truncation and the MPO bond-dimension truncation applied
#  after every multiplication. There is no Trotter/time-integration error.
#
#  Every intermediate rho_n computed on the way to rho_m is a legitimate
#  thermal state at beta_n = beta_unit*2^n, so it's recorded "for free"
#  (save_every_step=true by default) -- one run gives you a whole log-spaced
#  beta curve, not just the endpoint.
#
#  To cover beta values that don't land on THIS run's ladder, launch another
#  independent job with a different beta_unit (and/or m) -- e.g.
#      julia HC_Sz_xtrg.jl 0.01 10   # -> beta = 0.01, 0.02, 0.04, ..., 10.24
#      julia HC_Sz_xtrg.jl 0.015 10  # -> beta = 0.015, 0.03, ..., 15.36
#  These are fully independent processes and can run in parallel (e.g. as
#  separate SLURM array tasks). Each run writes its own CSV (tagged by
#  beta_unit/m in the filename) to avoid concurrent-write collisions; merge
#  the CSVs afterward if you want one combined table.
#
# ==============================================================================

function taylor_rho0(H::MPO, τ::Float64, sites;
                      order::Int=16, cutoff::Float64=1e-13, maxdim::Int=800)
    "Build rho0 = e^{-tau*H} as an MPO via a truncated Taylor series:
    rho0 = sum_{k=0}^{order} (-tau)^k/k! H^k, computed by the recursion
    term_k = (-tau/k)*H*term_{k-1}, term_0 = Id."

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

function mpo_trace(A::MPO, sites)
    "Tr(A) for an MPO A -- contracts the bra/ket physical indices directly
    (no ancilla needed). Uses dag() on the primed leg so QN-conserving indices
    have opposite arrow directions, as required for a valid contraction."

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


let
    Total_time = time()

#  -- Physical parameter setup ---
    N = 2
    M = 2
    Jnn, Jnnn, DMI, h = 1, 0.1, 0., 0.1
    ani = 0.

    obc_x = true
    obc_y = false
    Ox, Oy = "t", "f"

#  -- numerical setup ---
    QN_conservation = true

#  -- XTRG run parameters: this run reaches beta = beta_unit * 2^m --
    #  Pass beta_unit / m on the command line for parallel job submission,
    #  e.g.  julia HC_Sz_xtrg.jl 0.01 10   -- falls back to defaults below.
    beta_unit = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.01   # tau0 for this job
    m         = length(ARGS) >= 2 ? parse(Int,     ARGS[2]) : 10     # number of squarings
    beta_final = beta_unit * 2.0^m
    println("This run: beta_unit(tau0) = $beta_unit, m = $m  ->  beta_final = $beta_final")

    taylor_order    = 16
    rho_cutoff      = 1E-12
    rho_maxdim      = 400     # MPO bond dim for rho 
    save_every_step = true    # record beta_unit*2^n for every n=0..m (essentially free -- rho_n is already built on the way to rho_m)
    save_final_rho  = false    # also dump the final MPO + sites to disk via JLD2

#  -- files (tagged by beta_unit & m so parallel jobs never collide) --
    E_file = @sprintf(
        "../HC_data/%.i%.i_nnn%.2f_DM%.2f_h%.2f_Ox%s_Oy%s_tau%.4f_m%.i_k%.i_xtrg.csv",
        N, M, Jnnn, DMI, h, Ox, Oy, beta_unit, m, rho_maxdim
    )

# --- finite T (XTRG) ---

#  -- physical sites (NO ancilla needed for XTRG) --
    sites = siteinds("S=1/2", N*M*2; conserve_qns=QN_conservation)

    Hos, r = H_HC(N, M, Jnn, Jnnn, DMI, h, ani; anc=false, obc_x, obc_y)
    H = MPO(Hos, sites)
    M1OP = MPO(M1_op(N*M*2; anc=false), sites)
    M2OP = MPO(M2_op(N*M*2; anc=false), sites)

    function measure_and_write(rho_n::MPO, beta_n::Float64)
        Mz     = mpo_expect(rho_n, M1OP, sites)
        M2v    = mpo_expect(rho_n, M2OP, sites)
        E_n    = mpo_expect(rho_n, H, sites; cutoff=rho_cutoff, maxdim=rho_maxdim)
        dim    = maxlinkdim(rho_n)

        open(E_file, "a") do io
            d = @sprintf("%.6f,%.10f,%.10f,%.10f,%.i\n", beta_n, real(E_n), real(Mz), real(M2v), dim)
            write(io, d)
        end

        println("  beta = $beta_n   E = $E_n   chi = $dim")
    end

#  -- rho(tau0) via Taylor series --
    println("Building rho(tau0), tau0 = $beta_unit ...")
    rho_n = taylor_rho0(H, beta_unit, sites; order=taylor_order, cutoff=rho_cutoff, maxdim=rho_maxdim)
    Z = real(mpo_trace(rho_n, sites))
    rho_n = rho_n / Z
    beta_n = beta_unit

    save_every_step && measure_and_write(rho_n, beta_n)

#  -- repeated squaring: Eqs. (4)-(5) --
    for n in 1:m
        lp_time = time()

        rho_n = apply(rho_n, rho_n; cutoff=rho_cutoff, maxdim=rho_maxdim)
        Z = real(mpo_trace(rho_n, sites))
        rho_n = rho_n / Z
        beta_n *= 2

        println("Squaring step $n/$m -> beta = $beta_n, chi = $(maxlinkdim(rho_n))")
        println("Process peak RSS (GB): ", Sys.maxrss()/1.07E9)

        if save_every_step || n == m
            measure_and_write(rho_n, beta_n)
        end

        println("Loop time: $(time()-lp_time)")
        println(" ")
        GC.gc()
    end

#  -- save final rho / sites --
    if save_final_rho
        Psi_file = @sprintf("./HC_data/Psi/%.i%.i_DM%.2f_Ox%s_Oy%s_tau%.6f_m%.i_k%.i_rho_xtrg.jld2",
                             N, M, DMI, Ox, Oy, beta_unit, m, rho_maxdim)
        save_object(Psi_file, rho_n)
        Sites_file = @sprintf("./HC_data/Psi/%.i%.i_DM%.2f_Ox%s_Oy%s_tau%.6f_m%.i_k%.i_sites_xtrg.jld2",
                               N, M, DMI, Ox, Oy, beta_unit, m, rho_maxdim)
        save_object(Sites_file, sites)
    end

    println("Reached beta = $beta_n in $m squarings from tau0 = $beta_unit")
    println("Total computational time: $(time()-Total_time)")
end