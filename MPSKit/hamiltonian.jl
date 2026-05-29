using MPSKit, MPSKitModels, TensorKit

"""
    H_HC(n, m, J, j, D, h, ani; auc=false, obc_x=false, obc_y=false)

    Honeycomb Hamiltonian in MPSKit/MPSKitModels style.

    Parameters
    ----------
    n, m    : lattice dimensions (n rows, m unit cells per row)
    J       : nearest-neighbour (NN) exchange
    j       : next-nearest-neighbour (NNN) exchange
    D       : DM interaction strength (complex phase on NNN transverse terms)
    h       : Zeeman field (along z)
    ani     : single-ion easy-axis anisotropy (Sz²)
    anc     : if true, interleave ancilla sites for purification
    obc_x   : open boundary in x (inter-row direction)
    obc_y   : open boundary in y (intra-row direction)
"""

function merge_ops!(target, source)
    for (key, op) in source
        target[key] = get(target, key, zero(op)) + op
    end
end

function H_HC(QN_Sz, n, m, J, j, D, h, ani; kwargs...)
    return H_HC(Val(QN_Sz), n, m, J, j, D, h, ani; kwargs...)
end

function H_HC(QN_Sz::Val{false}, n, m, J, j, D, h, ani; anc=false, obc_x=false, obc_y=false)

    col = m * 2          # physical sites per row
    N   = n * col        # total physical sites
    X   = anc ? 2 : 1    # stride: 2 if ancilla interleaved, else 1
    N_total = N * X      # total sites on the MPS (including ancilla)


    # --- Build the MPSKit lattice ---
    # All sites are spin-1/2 (ℂ^2). Ancilla sites are identical in type.
    lattice = fill(ℂ^2, N_total)

    # --- Spin operators ---
    S_z = TensorMap([1/2 0; 0 -1/2], ℂ^2, ℂ^2)
    S_plus = TensorMap([0 1; 0 0], ℂ^2, ℂ^2)   # S⁺
    S_min = TensorMap([0 0; 1 0], ℂ^2, ℂ^2)   # S⁻

    Id = id(ℂ^2)

    mps_idx(i) = i * X    # physical site i → MPS site
    res(i, n) = mod(i - 1, n) + 1

    # --- Collect operator dicts ---
    local_operators      = Dict()
    nn_operators         = Dict()
    nnn_operators        = Dict()
    all_operators        = Dict()

    for a in 0:n-1
        for b in 1:m

            # ── A site ──────────────────────────────────────────────
            y_A  = 2b - 1
            i_A  = col * a + y_A
            ii_A = mps_idx(i_A)     # MPS index of A

            # ── B site ──────────────────────────────────────────────
            y_B  = 2b
            i_B  = col * a + y_B
            ii_B = mps_idx(i_B)     # MPS index of B

            # ════════════════════════════════════════════════════════
            # Single-site terms (Zeeman + anisotropy) — A and B
            # ════════════════════════════════════════════════════════
            for ii in (ii_A, ii_B)
                # Zeeman: -h Sz
                local_operators[(ii,)] = get(local_operators, (ii,), zero(S_z)) + (-h)*S_z

                # Anisotropy: -ani Sz² = -ani Sz⊗Sz on same site (single-site)
                local_operators[(ii,)] = local_operators[(ii,)] + (-ani)*S_z*S_z
            end

            # ════════════════════════════════════════════════════════
            # Helper to add a two-site Heisenberg-like bond
            #   H_bond = c_zz * Sz⊗Sz + c_pm * S+⊗S- + c_mp * S-⊗S+
            # ════════════════════════════════════════════════════════
            function add_2op!(dict, i1, i2, c_zz, c_pm, c_mp)
                i1, i2 = min(i1,i2), max(i1,i2)   # canonical order: lower index first
                key = (i1, i2)
                op = c_zz * (S_z ⊗ S_z) +
                     c_pm * (S_plus ⊗ S_min) +
                     c_mp * (S_min ⊗ S_plus)
                dict[key] = get(dict, key, zero(op)) + op
            end

            # ════════════════════════════════════════════════════════
            # NN bonds (three neighbours)
            # ════════════════════════════════════════════════════════

            # 1) A–B intra-cell bond (always present)
            add_2op!(nn_operators, ii_A, ii_B, 
                -J/2, -J/4, -J/4
            )
            # 2) A – previous site in row (y-1, periodic/open in y)
            O_ = (obc_y && y_A == 1) ? 0 : 1
            nb = mps_idx(col * a + res(y_A - 1, col))
            add_2op!(nn_operators, ii_A, nb, 
                -J*O_/2, -J*O_/4, -J*O_/4
            )
            # 3) A – cross-row neighbour (i+1-col, periodic/open in x)
            O_ = (obc_x && i_A+1-col < 1) ? 0 : 1
            nb = mps_idx(res(i_A + 1 - col, N))
            add_2op!(nn_operators, ii_A, nb, 
                -J*O_/2, -J*O_/4, -J*O_/4
            )

            # ════════════════════════════════════════════════════════
            # NNN bonds — A site (six neighbours, with ±iD phase)
            # ════════════════════════════════════════════════════════

            # +y direction in row
            O_ = (obc_y && y_A+2 > col) ? 0 : 1
            nb = mps_idx(col*a + res(y_A+2, col))
            add_2op!(nnn_operators, ii_A, nb,
                -j*O_/2, (-j + im*D)*O_/4, (-j - im*D)*O_/4
            )

            # -y direction in row
            O_ = (obc_y && y_A-2 < 1) ? 0 : 1
            nb = mps_idx(col*a + res(y_A-2, col))
            add_2op!(nnn_operators, ii_A, nb,
                -j*O_/2, (-j - im*D)*O_/4, (-j + im*D)*O_/4
            )

            # -x (previous row, same y)
            O_ = (obc_x && i_A-col < 1) ? 0 : 1
            nb = mps_idx(res(i_A-col, N))
            add_2op!(nnn_operators, ii_A, nb,
                -j*O_/2, (-j + im*D)*O_/4, (-j - im*D)*O_/4
            )

            # +x (next row, same y)
            O_ = (obc_x && i_A+col > N) ? 0 : 1
            nb = mps_idx(res(i_A+col, N))
            add_2op!(nnn_operators, ii_A, nb,
                -j*O_/2, (-j - im*D)*O_/4, (-j + im*D)*O_/4
            )

            # diagonal: next row, -y
            O_x = (obc_x && a == n-1) ? 0 : 1
            O_y = (obc_y && y_A-2 < 1) ? 0 : 1
            nb = mps_idx(res(col*(a+1) + res(y_A - 2, col), N))
            add_2op!(nnn_operators, ii_A, nb,
                (-j/2)*O_x*O_y, (-j + im*D)*O_x*O_y/4, (-j - im*D)*O_x*O_y/4
            )

            # diagonal: prev row, +y
            O_x = (obc_x && a == 0) ? 0 : 1
            O_y = (obc_y && y_A+2 > col) ? 0 : 1
            nb = mps_idx(res(col*(a-1) + res(y_A + 2, col), N))
            add_2op!(nnn_operators, ii_A, nb,
                (-j/2)*O_x*O_y, (-j - im*D)*O_x*O_y/4, (-j + im*D)*O_x*O_y/4
            )
            
            # ════════════════════════════════════════════════════════
            # NN bonds (three neighbours)
            # ════════════════════════════════════════════════════════

            # 1) A–B intra-cell bond (always present)
            add_2op!(nn_operators, ii_B, ii_A, 
                -J/2, -J/4, -J/4
            )
            # 2) B – previous site in row (y-1, periodic/open in y)
            O_ = (obc_y && y_B+1 > col) ? 0 : 1
            nb = mps_idx(col * a + res(y_B + 1, col))
            add_2op!(nn_operators, ii_B, nb, 
                -J*O_/2, -J*O_/4, -J*O_/4
            )
            # 3) A – cross-row neighbour (i+1-col, periodic/open in x)
            O_ = (obc_x && i_B-1+col > 1) ? 0 : 1
            nb = mps_idx(res(i_B-1+col, N))
            add_2op!(nn_operators, ii_B, nb, 
                -J*O_/2, -J*O_/4, -J*O_/4
            )

            # ════════════════════════════════════════════════════════
            # NNN bonds — B site (six neighbours, with ±iD phase)
            # ════════════════════════════════════════════════════════

            # -y direction in row
            O_ = (obc_y && y_B-2 < 1) ? 0 : 1
            nb = mps_idx(col * a + res(y_B - 2, col))
            add_2op!(nnn_operators, ii_B, nb,
                (-j/2)*O_, (-j + im*D)*O_/4, (-j - im*D)*O_/4
            )

            # +y direction in row
            O_ = (obc_y && y_B+2 > col) ? 0 : 1
            nb = mps_idx(col * a + res(y_B + 2, col))
            add_2op!(nnn_operators, ii_B, nb,
                (-j/2)*O_, (-j - im*D)*O_/4, (-j + im*D)*O_/4
            )

            # +x (next row)
            O_ = (obc_x && i_B+col > N) ? 0 : 1
            nb = mps_idx(res(i_B + col, N))
            add_2op!(nnn_operators, ii_B, nb,
                (-j/2)*O_, (-j + im*D)*O_/4, (-j - im*D)*O_/4
            )

            # -x (prev row)
            O_ = (obc_x && i_B-col < 1) ? 0 : 1
            nb = mps_idx(res(i_B - col, N))
            add_2op!(nnn_operators, ii_B, nb,
                (-j/2)*O_, (-j - im*D)*O_/4, (-j + im*D)*O_/4
            )

            # diagonal: prev row, +y
            O_x = (obc_x && a == 0) ? 0 : 1
            O_y = (obc_y && y_B+2 > col) ? 0 : 1
            nb = mps_idx(res(col*(a-1) + res(y_B + 2, col), N))
            add_2op!(nnn_operators, ii_B, nb,
                (-j/2)*O_x*O_y, (-j + im*D)*O_x*O_y/4, (-j - im*D)*O_x*O_y/4
            )

            # diagonal: next row, -y
            O_x = (obc_x && a == n-1) ? 0 : 1
            O_y = (obc_y && y_B-2 < 1) ? 0 : 1
            nb = mps_idx(res(col*(a+1) + res(y_B - 2, col), N))
            add_2op!(nnn_operators, ii_B, nb,
                (-j/2)*O_x*O_y, (-j - im*D)*O_x*O_y/4, (-j + im*D)*O_x*O_y/4
            )

        end
    end

    # ════════════════════════════════════════════════════════════════
    # Assemble Hamiltonian
    # ════════════════════════════════════════════════════════════════

    merge_ops!(all_operators, local_operators)
    merge_ops!(all_operators, nn_operators)
    merge_ops!(all_operators, nnn_operators)

    # Single construction call — MPSKit sees all range scales at once
    # and builds one consistent MPO bond space
    H = FiniteMPOHamiltonian(lattice, all_operators)

    return H
end

function H_HC(QN_Sz::Val{true}, n, m, J, j, D, h, ani; anc=false, obc_x=false, obc_y=false)

    col = m * 2          # physical sites per row
    N   = n * col        # total physical sites
    X   = anc ? 2 : 1    # stride: 2 if ancilla interleaved, else 1
    N_total = N * X      # total sites on the MPS (including ancilla)

    # -- Sz conservation --
    V_phys = U1Space(1 => 1, -1 => 1)   # replaces ℂ^2
    V_2    = V_phys ⊗ V_phys   # two-site physical space

    # --- Build the MPSKit lattice ---
    # All sites are spin-1/2 (ℂ^2). Ancilla sites are identical in type.
    lattice = fill(V_phys, N_total)

    # --- Spin operators ---
    # ── Symmetry-preserving physical space ───────────────────────────
    S_z = TensorMap([1/2 0; 0 -1/2], V_phys ← V_phys)
    S_z2 = TensorMap([1/4 0; 0 1/4], V_phys ← V_phys)
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
                    
    Id = id(ℂ^2)

    mps_idx(i) = i * X    # physical site i → MPS site
    res(i, n) = mod(i - 1, n) + 1

    # --- Collect operator dicts ---
    local_operators      = Dict()
    nn_operators         = Dict()
    nnn_operators        = Dict()
    all_operators        = Dict()

    for a in 0:n-1
        for b in 1:m

            # ── A site ──────────────────────────────────────────────
            y_A  = 2b - 1
            i_A  = col * a + y_A
            ii_A = mps_idx(i_A)     # MPS index of A

            # ── B site ──────────────────────────────────────────────
            y_B  = 2b
            i_B  = col * a + y_B
            ii_B = mps_idx(i_B)     # MPS index of B

            # ════════════════════════════════════════════════════════
            # Single-site terms (Zeeman + anisotropy) — A and B
            # ════════════════════════════════════════════════════════
            for ii in (ii_A, ii_B)
                # Zeeman: -h Sz
                local_operators[(ii,)] = get(local_operators, (ii,), zero(S_z)) + (-h)*S_z

                # Anisotropy: -ani Sz² = -ani Sz⊗Sz on same site (single-site)
                local_operators[(ii,)] = local_operators[(ii,)] + (-ani)*S_z*S_z
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
                -J, -J/2, -J/2
            )
            # 2) A – previous site in row (y-1, periodic/open in y)
            O_ = (obc_y && y_A == 1) ? 0 : 1
            nb = mps_idx(col * a + res(y_A - 1, col))
            add_2op!(nn_operators, ii_A, nb, 
                -J*O_, -J*O_/2, -J*O_/2
            )
            # 3) A – cross-row neighbour (i+1-col, periodic/open in x)
            O_ = (obc_x && i_A+1-col < 1) ? 0 : 1
            nb = mps_idx(res(i_A + 1 - col, N))
            add_2op!(nn_operators, ii_A, nb, 
                -J*O_, -J*O_/2, -J*O_/2
            )

            # ════════════════════════════════════════════════════════
            # NNN bonds — A site (six neighbours, with ±iD phase)
            # ════════════════════════════════════════════════════════

            # +y direction in row
            O_ = (obc_y && y_A+2 > col) ? 0 : 1
            nb = mps_idx(col*a + res(y_A+2, col))
            add_2op!(nnn_operators, ii_A, nb,
                -j*O_/2, (-j + im*D)*O_/4, (-j - im*D)*O_/4
            )

            # -y direction in row
            O_ = (obc_y && y_A-2 < 1) ? 0 : 1
            nb = mps_idx(col*a + res(y_A-2, col))
            add_2op!(nnn_operators, ii_A, nb,
                -j*O_/2, (-j - im*D)*O_/4, (-j + im*D)*O_/4
            )

            # -x (previous row, same y)
            O_ = (obc_x && i_A-col < 1) ? 0 : 1
            nb = mps_idx(res(i_A-col, N))
            add_2op!(nnn_operators, ii_A, nb,
                -j*O_/2, (-j + im*D)*O_/4, (-j - im*D)*O_/4
            )

            # +x (next row, same y)
            O_ = (obc_x && i_A+col > N) ? 0 : 1
            nb = mps_idx(res(i_A+col, N))
            add_2op!(nnn_operators, ii_A, nb,
                -j*O_/2, (-j - im*D)*O_/4, (-j + im*D)*O_/4
            )

            # diagonal: next row, -y
            O_x = (obc_x && a == n-1) ? 0 : 1
            O_y = (obc_y && y_A-2 < 1) ? 0 : 1
            nb = mps_idx(res(col*(a+1) + res(y_A - 2, col), N))
            add_2op!(nnn_operators, ii_A, nb,
                (-j/2)*O_x*O_y, (-j + im*D)*O_x*O_y/4, (-j - im*D)*O_x*O_y/4
            )

            # diagonal: prev row, +y
            O_x = (obc_x && a == 0) ? 0 : 1
            O_y = (obc_y && y_A+2 > col) ? 0 : 1
            nb = mps_idx(res(col*(a-1) + res(y_A + 2, col), N))
            add_2op!(nnn_operators, ii_A, nb,
                (-j/2)*O_x*O_y, (-j - im*D)*O_x*O_y/4, (-j + im*D)*O_x*O_y/4
            )

            # ════════════════════════════════════════════════════════
            # NNN bonds — B site (six neighbours, with ±iD phase)
            # ════════════════════════════════════════════════════════

            # -y direction in row
            O_ = (obc_y && y_B-2 < 1) ? 0 : 1
            nb = mps_idx(col * a + res(y_B - 2, col))
            add_2op!(nnn_operators, ii_B, nb,
                (-j/2)*O_, (-j + im*D)*O_/4, (-j - im*D)*O_/4
            )

            # +y direction in row
            O_ = (obc_y && y_B+2 > col) ? 0 : 1
            nb = mps_idx(col * a + res(y_B + 2, col))
            add_2op!(nnn_operators, ii_B, nb,
                (-j/2)*O_, (-j - im*D)*O_/4, (-j + im*D)*O_/4
            )

            # +x (next row)
            O_ = (obc_x && i_B+col > N) ? 0 : 1
            nb = mps_idx(res(i_B + col, N))
            add_2op!(nnn_operators, ii_B, nb,
                (-j/2)*O_, (-j + im*D)*O_/4, (-j - im*D)*O_/4
            )

            # -x (prev row)
            O_ = (obc_x && i_B-col < 1) ? 0 : 1
            nb = mps_idx(res(i_B - col, N))
            add_2op!(nnn_operators, ii_B, nb,
                (-j/2)*O_, (-j - im*D)*O_/4, (-j + im*D)*O_/4
            )

            # diagonal: prev row, +y
            O_x = (obc_x && a == 0) ? 0 : 1
            O_y = (obc_y && y_B+2 > col) ? 0 : 1
            nb = mps_idx(res(col*(a-1) + res(y_B + 2, col), N))
            add_2op!(nnn_operators, ii_B, nb,
                (-j/2)*O_x*O_y, (-j + im*D)*O_x*O_y/4, (-j - im*D)*O_x*O_y/4
            )

            # diagonal: next row, -y
            O_x = (obc_x && a == n-1) ? 0 : 1
            O_y = (obc_y && y_B-2 < 1) ? 0 : 1
            nb = mps_idx(res(col*(a+1) + res(y_B - 2, col), N))
            add_2op!(nnn_operators, ii_B, nb,
                (-j/2)*O_x*O_y, (-j - im*D)*O_x*O_y/4, (-j + im*D)*O_x*O_y/4
            )

        end
    end

    # ════════════════════════════════════════════════════════════════
    # Assemble Hamiltonian
    # ════════════════════════════════════════════════════════════════

    merge_ops!(all_operators, local_operators)
    merge_ops!(all_operators, nn_operators)
    merge_ops!(all_operators, nnn_operators)

    # Single construction call — MPSKit sees all range scales at once
    # and builds one consistent MPO bond space
    H = FiniteMPOHamiltonian(lattice, all_operators)

    return H
end
