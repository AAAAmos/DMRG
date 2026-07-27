
function make_shifted_spaces(col, χ, q_tilde)
    # q_tilde = 2 * <Sz> per site (integer)
    # e.g. fully polarised ↑: q_tilde = +1 (one ↑ per site, charge = +1)
    #      Sz=0 (standard):   q_tilde =  0
    #      fully polarised ↓: q_tilde = -1

    # Shifted physical space: original charges {+1, -1} shifted by -q_tilde
    V_phys_shifted = U1Space((1 - q_tilde) => 1, (-1 - q_tilde) => 1)

    # Virtual space: now uniform across ALL bonds (net charge per unit cell = 0)
    # Range needed: at bond k, charge can range ±k*max(|s̃|)
    s_max = max(abs(1 - q_tilde), abs(-1 - q_tilde))
    q_max = round(Int, s_max * col)
    # V_virt = U1Space(Dict(q => χ for q in -q_max:2:q_max))
    V_virt = U1Space(Dict(q => χ for q in 0:2:q_max))
    # V_virt = U1Space(Dict(q => χ for q in 0:2:10))
    N_qnsectors = length(0:2:q_max)

    return V_phys_shifted, V_virt, N_qnsectors
end

function bond_dim_str(ψ::InfiniteMPS)
    # ── Bond-dimension monitor ────────────────────────────────────────────
    # Returns "max=D  [c→d, ...]" using bond 1 as the representative sector layout.
    max_bd = maximum(dim(right_virtualspace(ψ, n)) for n in 1:length(ψ))
    V = right_virtualspace(ψ, 1)
    sec_str = join((@sprintf("%s→%d", c, dim(V, c)) for c in sectors(V)), ", ")
    return @sprintf("max=%d  [%s]", max_bd, sec_str)
end

function excitation_entanglement_center(ϕ::LeftGaugedQP)
    
    # ── Entanglement spectrum of excited states ───────────────────────────────
    # Algorithm: left-canonicalize all B tensors in the unit cell of ϕ.
    #   - Start from ψ.C[0]: the ground-state Schmidt decomposition at the left
    #     unit-cell seam; this is the correct left boundary condition.
    #   - At each site n, absorb the running center C into B[n] from the left,
    #     then QR-decompose to restore left-orthogonality and push C rightward.
    #   - The aux (sector utility) leg is moved to codomain via permute so that
    #     C stays a pure virtual bond matrix (domain = virt_R) after every step.
    #   - The final C at the right seam encodes entanglement between the
    #     left-canonical unit cell (tied to the left fixed point) and the right
    #     fixed point (the semi-infinite AR chain).

    ψ = ϕ.left_gs
    N = length(ψ)

    C = ψ.C[0]   # left boundary: ground-state center at bond 0

    for n in 1:N
        B = ϕ[n]   # B[-1 -2; -3 -4] = (virt_L, phys ; aux, virt_R)

        # Move aux from domain to codomain so C remains a virtual bond matrix.
        B_perm = permute(B, ((1, 2, 3), (4,)))   # (virt_L, phys, aux' ; virt_R)

        # Absorb running center from left (contracts domain of C with virt_L of B).
        @plansor AC[-1 -2 -3; -4] := C[-1; 1] * B_perm[1 -2 -3; -4]

        # QR: left-orthogonalize (virt, phys, aux') and propagate C rightward.
        _, C = left_orth(AC)
    end

    # C is the orthogonality center at the right unit-cell seam.
    # Its singular values give the entanglement spectrum at this cut.
    return C
end

function gauge_fix_mps(ψ::InfiniteMPS)
    # Put every C[n] in Schmidt (diagonal) gauge so that excitation B tensors
    # are computed in a canonical, run-to-run reproducible gauge.
    #
    # For each bond n, C[n] = U·S·Vd.  We absorb:
    #   U  into the RIGHT virtual of AL at the site LEFT  of bond n
    #   Vd into the LEFT  virtual of AR at the site RIGHT of bond n
    # leaving C[n] = S (diagonal, positive, decreasing).
    #
    # Note: AL[n] has layout (virt_L, phys ; virt_R) and
    #       AR[n] has layout (virt_L, phys ; virt_R) in MPSKit convention.
    # Verify that domain(AL[n]) == codomain(U) before using in production.

    N     = length(ψ)
    new_AL = collect(ψ.AL)   # mutable copies
    new_AR = collect(ψ.AR)
    new_C  = [ψ.C[n] for n in 0:N-1]

    for n in 0:N-1
        U, S, Vd = svd_compact(ψ.C[n])   # C[n] = U * S * Vd

        # Site to the LEFT of bond n (1-indexed, wraps for bond 0)
        l = n == 0 ? N : n
        # Site to the RIGHT of bond n
        r = n + 1

        # AL[l]: absorb U on the right-virtual leg (domain of AL = virt_R of site l)
        @plansor new_AL[l][-1 -2; -3] := ψ.AL[l][-1 -2; 1] * U[1; -3]

        # AR[r]: absorb Vd on the left-virtual leg (first codomain leg of AR = virt_L of site r)
        @plansor new_AR[r][-1 -2; -3] := Vd[-1; 1] * ψ.AR[r][1 -2; -3]

        new_C[n+1] = S   # 1-indexed storage; C[n] → S
    end

    return InfiniteMPS(new_AL, PeriodicVector(new_C), new_AR)
end

function merge_ops!(target, source)
    for (key, op) in source
        target[key] = get(target, key, zero(op)) + op
    end
end

function H_HC_uMPS(ny, J, j, D, h, ani, V_phys; anc=false, obc_y=false)

    """ 
        Construct a uniform Hamiltonian for UMPS.
        unit cell length = 2ny.
        infinite length in x direction.
    """

    N = 2ny            # total physical sites
    X = anc ? 2 : 1    # stride: 2 if ancilla interleaved, else 1

    V_2    = V_phys ⊗ V_phys

    # --- Build the MPSKit lattice ---
    lattice = PeriodicVector([V_phys for _ in 1:N])

    # --- Spin operators ---
    # ── Symmetry-preserving physical space ───────────────────────────
    S_z = TensorMap([1/2 0; 0 -1/2], V_phys ← V_phys)
    SzSz = TensorMap([1/4 0 0 0;
                    0 -1/4 0 0;
                    0  0 -1/4 0;
                    0  0  0 1/4], V_2 ← V_2)

    # S+S-: site 1 raises (+1), site 2 lowers (-1) → net charge 0
    SpSm = TensorMap([0 0 0 0;
                    0 0 1 0;
                    0 0 0 0;
                    0 0 0 0], V_2 ← V_2)

    # S-S+: site 1 lowers (-1), site 2 raises (+1) → net charge 0
    SmSp = TensorMap([0 0 0 0;
                    0 0 0 0;
                    0 1 0 0;
                    0 0 0 0], V_2 ← V_2)

    mps_idx(i) = i * X    # physical site i → MPS site
    res(i, n) = mod(i - 1, n) + 1

    # --- Collect operator dicts ---
    local_operators      = Dict()
    nn_operators         = Dict()
    nnn_operators        = Dict()
    all_operators        = Dict()

    for b in 1:ny

        # ── A site ──────────────────────────────────────────────
        i_A  = 2b - 1
        ii_A = mps_idx(i_A)     # MPS index of A

        # ── B site ──────────────────────────────────────────────
        i_B  = 2b
        ii_B = mps_idx(i_B)     # MPS index of B

        # ════════════════════════════════════════════════════════
        # Single-site terms (Zeeman + anisotropy) — A and B
        # ════════════════════════════════════════════════════════
        for ii in (ii_A, ii_B)
            # Zeeman: -h Sz
            local_operators[(ii,)] = get(local_operators, (ii,), zero(S_z)) + (-h)*S_z

            # Anisotropy: -ani Sz² = -ani Sz⊗Sz on same site (single-site)
            # local_operators[(ii,)] = local_operators[(ii,)] + (-ani)*S_z*S_z
        end

        # ════════════════════════════════════════════════════════
        # Helper to add a two-site Heisenberg-like bond
        #   H_bond = c_zz * Sz⊗Sz + c_pm * S+⊗S- + c_mp * S-⊗S+
        # ════════════════════════════════════════════════════════
        function add_2op!(dict, i1, i2, c_zz, c_pm, c_mp)
            i1, i2 = min(i1, i2), max(i1, i2)
            key = (i1, i2)
            op  = c_zz*SzSz + c_pm*SpSm + c_mp*SmSp
            dict[key] = get(dict, key, zero(op)) + op
        end

        # ════════════════════════════════════════════════════════
        # NN bonds (three neighbours)
        # ════════════════════════════════════════════════════════

        # 1) A–B intra-cell bond (always present)
        add_2op!(nn_operators, ii_A, ii_B, 
            -J/2, -J/4, -J/4
        )
        # println("nn: $ii_A - $ii_B")
        # 2) A – previous site in row (y-1, periodic/open in y)
        O_ = (obc_y && i_A == 1) ? 0 : 1
        nb = mps_idx(res(i_A - 1, N))
        add_2op!(nn_operators, ii_A, nb, 
            -J*O_/2, -J*O_/4, -J*O_/4
        )
        # println("nn: $ii_A - $nb, $O_")
        # 3) A – previous-row neighbour 
        nb = mps_idx(i_A + 1 - N)
        add_2op!(nn_operators, ii_A, nb, 
            -J/2, -J/4, -J/4
        )
        # println("nn: $ii_A - $nb")

        # ════════════════════════════════════════════════════════
        # NNN bonds — A site (six neighbours, with ±iD phase)
        # ════════════════════════════════════════════════════════

        # +y direction in row
        O_ = (obc_y && i_A+2 > N) ? 0 : 1
        nb = mps_idx(res(i_A+2, N))
        add_2op!(nnn_operators, ii_A, nb,
            -j*O_/2, (-j + im*D)*O_/4, (-j - im*D)*O_/4
        )
        # println("nnn: $ii_A - $nb, $O_")

        # -y direction in row
        O_ = (obc_y && i_A-2 < 1) ? 0 : 1
        nb = mps_idx(res(i_A-2, N))
        add_2op!(nnn_operators, ii_A, nb,
            -j*O_/2, (-j - im*D)*O_/4, (-j + im*D)*O_/4
        )
        # println("nnn: $ii_A - $nb, $O_")

        # -x (previous row, same y)
        nb = mps_idx(i_A-N)
        add_2op!(nnn_operators, ii_A, nb,
            -j/2, (-j + im*D)/4, (-j - im*D)/4
        )
        # println("nnn: $ii_A - $nb")

        # +x (next row, same y)
        nb = mps_idx(i_A+N)
        add_2op!(nnn_operators, ii_A, nb,
            -j/2, (-j - im*D)/4, (-j + im*D)/4
        )
        # println("nnn: $ii_A - $nb")

        # diagonal: next row, -y
        O_y = (obc_y && i_A-2 < 1) ? 0 : 1
        nb = mps_idx(res(i_A - 2, N) + N)
        add_2op!(nnn_operators, ii_A, nb,
            (-j/2)*O_y, (-j + im*D)*O_y/4, (-j - im*D)*O_y/4
        )
        # println("nnn: $ii_A - $nb, $O_y")

        # diagonal: prev row, +y
        O_y = (obc_y && i_A+2 > N) ? 0 : 1
        nb = mps_idx(res(i_A + 2, N) - N)
        add_2op!(nnn_operators, ii_A, nb,
            (-j/2)*O_y, (-j - im*D)*O_y/4, (-j + im*D)*O_y/4
        )
        # println("nnn: $ii_A - $nb, $O_y")
        
        # ════════════════════════════════════════════════════════
        # NN bonds (three neighbours)
        # ════════════════════════════════════════════════════════

        # 1) A–B intra-cell bond (always present)
        add_2op!(nn_operators, ii_B, ii_A, 
            -J/2, -J/4, -J/4
        )
        # println("nn: $ii_A - $ii_B")
        # 2) B – previous site in row (y-1, periodic/open in y)
        O_ = (obc_y && i_B + 1 > N) ? 0 : 1
        nb = mps_idx(res(i_B + 1, N))
        add_2op!(nn_operators, ii_B, nb, 
            -J*O_/2, -J*O_/4, -J*O_/4
        )
        # println("nn: $ii_A - $nb, $O_")
        # 3) B – previous-row neighbour 
        nb = mps_idx(i_B - 1 + N)
        add_2op!(nn_operators, ii_B, nb, 
            -J/2, -J/4, -J/4
        )
        # println("nn: $ii_A - $nb")

        # ════════════════════════════════════════════════════════
        # NNN bonds — B site (six neighbours, with ±iD phase)
        # ════════════════════════════════════════════════════════

        # -y direction in row
        O_ = (obc_y && i_B-2 < 1) ? 0 : 1
        nb = mps_idx(res(i_B - 2, N))
        add_2op!(nnn_operators, ii_B, nb,
            (-j/2)*O_, (-j + im*D)*O_/4, (-j - im*D)*O_/4
        )
        # println("nnn: $ii_B - $nb, $O_")

        # +y direction in row
        O_ = (obc_y && i_B+2 > N) ? 0 : 1
        nb = mps_idx(res(i_B + 2, N))
        add_2op!(nnn_operators, ii_B, nb,
            (-j/2)*O_, (-j - im*D)*O_/4, (-j + im*D)*O_/4
        )
        # println("nnn: $ii_B - $nb, $O_")

        # +x (next row)
        nb = mps_idx(i_B + N)
        add_2op!(nnn_operators, ii_B, nb,
            (-j/2), (-j + im*D)/4, (-j - im*D)/4
        )
        # println("nnn: $ii_B - $nb")

        # -x (prev row)
        nb = mps_idx(i_B - N)
        add_2op!(nnn_operators, ii_B, nb,
            (-j/2), (-j - im*D)/4, (-j + im*D)/4
        )
        # println("nnn: $ii_B - $nb")

        # diagonal: prev row, +y
        O_y = (obc_y && i_B+2 > N) ? 0 : 1
        nb = mps_idx(res(i_B + 2, N) - N)
        add_2op!(nnn_operators, ii_B, nb,
            (-j/2)*O_y, (-j + im*D)*O_y/4, (-j - im*D)*O_y/4
        )
        # println("nnn: $ii_B - $nb, $O_y")

        # diagonal: next row, -y
        O_y = (obc_y && i_B-2 < 1) ? 0 : 1
        nb = mps_idx(res(i_B - 2, N) + N)
        add_2op!(nnn_operators, ii_B, nb,
            (-j/2)*O_y, (-j - im*D)*O_y/4, (-j + im*D)*O_y/4
        )
        # println("nnn: $ii_B - $nb, $O_y")

    end
    
    # ════════════════════════════════════════════════════════════════
    # Assemble Hamiltonian
    # ════════════════════════════════════════════════════════════════
    merge_ops!(all_operators, local_operators)
    merge_ops!(all_operators, nn_operators)
    merge_ops!(all_operators, nnn_operators)

    # Single construction call — MPSKit sees all range scales at once
    # and builds one consistent MPO bond space
    H = InfiniteMPOHamiltonian(lattice, all_operators)

    return H
end