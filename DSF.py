import gc
import resource

from quspin.basis import spin_basis_general
from quspin.operators import (
        hamiltonian,
    )
from quspin.tools.lanczos import (
        lanczos_full
    )
    
import numpy as np
from time import time
# import scipy as sp
# from multiprocessing import Pool

def bootstrap_mean(O_r, Id_r, n_bootstrap=100):
    """
    Uses boostraping to esimate the error due to sampling.

    O_r: numerator
    Id_r: denominator
    n_bootstrap: bootstrap sample size

    """
    O_r = np.asarray(O_r)
    Id_r = np.asarray(Id_r)
    #
    avg = np.nanmean(O_r, axis=0) / np.nanmean(Id_r, axis=0)
    n_Id = Id_r.shape[0]
    # n_N = O_r.shape[0]
    #
    i_iter = (np.random.randint(n_Id, size=n_Id) for i in range(n_bootstrap))
    #
    bootstrap_iter = (
        np.nanmean(O_r[i, ...], axis=0) / np.nanmean(Id_r[i, ...], axis=0)
        for i in i_iter
    )
    diff_iter = ((bootstrap - avg) ** 2 for bootstrap in bootstrap_iter)
    err = np.sqrt(sum(diff_iter) / n_bootstrap)
    #
    return avg, err

def build_basis(n, s):
    ### compute basis
    basis = spin_basis_general(n, S = s, pauli=0)
    print("Hilbert space size: {0:d}.\n".format(basis.Ns))
    
    r = range(n) # for F.T.
    
    return basis, r

def build_HC_basis(m, n):

  ### compute basis
  m_2 = m*2
  N = m_2*n
  print("constructed hexagonal lattice with {0:d} sites.\n".format(N))

  basis = spin_basis_general(N, S = "1/2", pauli = 0)
  print("Hilbert space size: {0:d}.\n".format(basis.Ns))
  
  r = []
  
  for i in range(n):
    for j in range(m):
      r.append([(3*i + 2)/2, 3**0.5*(-i/2 + j) ])
      r.append([(3*i + 1)/2, 3**0.5*(-i/2 + j + 0.5)])
  
  return basis, r

def H_BLBQ(N, theta, basis, obc=False):

    #### set up Heisenberg Hamiltonian with quspin #####

    # set up spin-spin interaction lists
    J1 = np.cos(theta)
    J2 = np.sin(theta)
    # J1 = 1
    # J2 = 1

    zz = []
    pn = []
    np_ = []
    
    zzzz, zzpn, zznp = [], [], []
    pnzz, pnpn, pnnp = [], [], []
    npzz, nppn, npnp = [], [], []
    
    if obc: n = N - 1
    else: n = N
    
    
    for i in range(n):
        j = (i+1)%N
    
        zz.append([J1, i, j])
        pn.append([J1/2, i, j])
        np_.append([J1/2, i, j])
        
        zzzz.append([J2, i, j, i, j])
        zzpn.append([J2/2, i, j, i, j])
        zznp.append([J2/2, i, j, i, j])
        
        pnzz.append([J2/2, i, j, i, j])
        pnpn.append([J2/4, i, j, i, j])
        pnnp.append([J2/4, i, j, i, j])
        
        npzz.append([J2/2, i, j, i, j])
        nppn.append([J2/4, i, j, i, j])
        npnp.append([J2/4, i, j, i, j])
  
    # define spin-spin interaction lists 
    static = [["+-", pn], ["-+", pn], ["zz", zz],
        ["zzzz", zzzz], ["zz+-", zzpn], ["zz-+", zznp],
        ["+-zz", pnzz], ["+-+-", pnpn], ["+--+", pnnp],
        ["-+zz", npzz], ["-++-", nppn], ["-+-+", npnp]
        ]
    
    dynamic = []
    
    ### construct Hamiltonian
    H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
    return H


def H_heisenburg(N, g, basis, obc=False):

    #### set up Heisenberg Hamiltonian with quspin #####

    # set up spin-spin interaction lists
    J1 = 1

    zz, pn, z = [], [], []
    
    if obc: n = N - 1
    else: n = N
    
    for i in range(n):
        j = (i+1)%N
    
        zz.append([J1, i, j])
        pn.append([J1/2, i, j])
        z.append([g, i])
  
    # define spin-spin interaction lists 
    static = [["+-", pn], ["-+", pn], ["zz", zz], ["z", z]]
    dynamic = []
    
    ### construct Hamiltonian
    H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
    return H


def H_HC(m, n, Jxy, Jz, J_nnn, J_dmi, h, basis):

  #### set up Heisenberg Hamiltonian with quspin #####

  m_2 = m*2
  N = m_2*n
  
  # set up spin-spin interaction lists
  
  Sz = []
  SzSz = []
  SpSn = []
  SnSp = []
  
  for i in range(n):
    for j in range(m):
    
      #Zeamann term
      
      Sz.append([-h, m_2*i + 2*j])
      Sz.append([-h, m_2*i + 2*j+1])
      
      #N.N. coupling
      
      SzSz.append([-Jz, m_2*i + 2*j, m_2*i + (2*j+1)])
      SpSn.append([-Jxy/2, m_2*i + 2*j, m_2*i + (2*j+1)])
      SnSp.append([-Jxy/2, m_2*i + 2*j, m_2*i + (2*j+1)])
  
      SzSz.append([-Jz, m_2*i + 2*j, m_2*i + (2*j-1)%m_2])
      SpSn.append([-Jxy/2, m_2*i + 2*j, m_2*i + (2*j-1)%m_2])
      SnSp.append([-Jxy/2, m_2*i + 2*j, m_2*i + (2*j-1)%m_2])
      
      SzSz.append([-Jz, m_2*i + 2*j, (m_2*(i+1) + 2*j+1)%N])
      SpSn.append([-Jxy/2, m_2*i + 2*j, (m_2*(i+1) + 2*j+1)%N])
      SnSp.append([-Jxy/2, m_2*i + 2*j, (m_2*(i+1) + 2*j+1)%N])
      
      #N.N.N. coupling and DMI for A sites
      
      SzSz.append([-J_nnn, m_2*i + 2*j, (m_2*(i+1) + 2*j)%N])
      SpSn.append([-J_nnn/2 + J_dmi, m_2*i + 2*j, (m_2*(i+1) + 2*j)%N])
      SnSp.append([-J_nnn/2 - J_dmi, m_2*i + 2*j, (m_2*(i+1) + 2*j)%N])   
        
      SzSz.append([-J_nnn, m_2*i + 2*j, m_2*i + (2*j+2)%m_2])
      SpSn.append([-J_nnn/2 + J_dmi, m_2*i + 2*j, m_2*i + (2*j+2)%m_2])
      SnSp.append([-J_nnn/2 - J_dmi, m_2*i + 2*j, m_2*i + (2*j+2)%m_2])   

      SzSz.append([-J_nnn, m_2*i + 2*j, (m_2*(i-1) + (2*j-2)%m_2)%N])
      SpSn.append([-J_nnn/2 + J_dmi, m_2*i + 2*j, (m_2*(i-1) + (2*j-2)%m_2)%N])
      SnSp.append([-J_nnn/2 - J_dmi, m_2*i + 2*j, (m_2*(i-1) + (2*j-2)%m_2)%N])   

      #N.N.N. coupling and DMI for B sites
      
      SzSz.append([-J_nnn, m_2*i + 2*j+1, (m_2*(i+1) + (2*j+3)%m_2)%N])
      SpSn.append([-J_nnn/2 + J_dmi, m_2*i + 2*j+1, (m_2*(i+1) + (2*j+3)%m_2)%N])
      SnSp.append([-J_nnn/2 - J_dmi, m_2*i + 2*j+1, (m_2*(i+1) + (2*j+3)%m_2)%N])   

      SzSz.append([-J_nnn, m_2*i + 2*j+1, m_2*i + (2*j-1)%m_2])
      SpSn.append([-J_nnn/2 + J_dmi, m_2*i + 2*j+1, m_2*i + (2*j-1)%m_2])
      SnSp.append([-J_nnn/2 - J_dmi, m_2*i + 2*j+1, m_2*i + (2*j-1)%m_2])   

      SzSz.append([-J_nnn, m_2*i + 2*j+1, (m_2*(i-1) + 2*j+1)%N])
      SpSn.append([-J_nnn/2 + J_dmi, m_2*i + 2*j+1, (m_2*(i-1) + 2*j+1)%N])
      SnSp.append([-J_nnn/2 - J_dmi, m_2*i + 2*j+1, (m_2*(i-1) + 2*j+1)%N])

  # define spin-spin interaction lists 
  static = [["+-", SpSn], ["-+", SnSp], ["zz", SzSz], ["z", Sz]]#
  dynamic = []
  
  ### construct Hamiltonian
  H = hamiltonian(static, dynamic, basis=basis, dtype=np.complex128, check_herm = False)
  return H

def delta(x):
    e = 0.05
    D = e/(e**2 + x**2) * 1/np.pi
    return D

def dsf_xt_exact(A, B, H, T, beta):

    E1, V1 = H.eigh()
    print(f"E0 = {E1[0]}")
    E1 -= E1[0]
    print(f"Eigenenergy: {E1}")
    
    A_M = A.tocsr()
    B_M = B.tocsr()
    
    V1dag = V1.conj().T
    V0 = V1[:, 0]
    V0dag = V0.conj().T

    pIp = (V1dag @ V1)  #(Ns, Ns)
    pAp = (V1dag @ A_M @ V1)  #(Ns, Ns)
    pBp = (V1dag @ B_M @ V1)  #(Ns, Ns)
    
    p0Ap = np.squeeze(V0dag @ A_M @ V1)  #(1, Ns)
    pBp0 = np.squeeze(V1dag @ B_M @ V0)  #(Ns, 1)

    delta_E = E1[:, None] - E1[None, :]
    
    p = np.exp(-np.outer(beta, E1))
    Z = np.einsum("...j->...", p) 
    
    results_FT = np.zeros((len(T), len(beta)), dtype=np.complex128)
    results_T0 = np.zeros(len(T), dtype=np.complex128)
    M_expect = np.zeros(len(beta), dtype=np.complex128)
    
    M_expect[:] = np.einsum('bi,ij,ji->b', p, pIp, pBp)
    
    for t in range(len(T)):
        
        pAp1 = np.einsum('...i,ij,ij->...ij', p, pAp , np.exp(1j*delta_E*T[t])) 
        pABp = np.einsum('...ij,ji->...', pAp1, pBp)

        results_T0[t] = np.einsum('i,i->', p0Ap, (pBp0 * np.exp(-1j*E1*T[t])))
        results_FT[t, :] = pABp
            
        print("Peak memory used for exact solution w={0:.1f}:{1:.4f} MB".format(T[t], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    
    return E1, V1, results_FT, Z, results_T0, M_expect

def dsf_exact(A, B, H, w, beta):

    E1, V1 = H.eigh()
    print(f"E0 = {E1[0]}")
    E1 -= E1[0]
    # print(f"Eigenenergy: {E1}")
    
    A_M = A.tocsr()
    B_M = B.tocsr()
    
    V1dag = V1.conj().T
    V0 = V1[:, 0]
    V0dag = V0.conj().T

    pIp = (V1dag @ V1)  #(Ns, Ns)
    pAp = (V1dag @ A_M @ V1)  #(Ns, Ns)
    pBp = (V1dag @ B_M @ V1)  #(Ns, Ns)
    
    p0Ap = np.squeeze(V0dag @ A_M @ V1)  #(1, Ns)
    pBp0 = np.squeeze(V1dag @ B_M @ V0)  #(Ns, 1)

    delta_E = E1[:, None] - E1[None, :]
    
    p = np.exp(-np.outer(beta, E1))
    Z = np.einsum("...j->...", p) 
    
    results_FT = np.zeros((len(w), len(beta)), dtype=np.complex128)
    results_T0 = np.zeros(len(w), dtype=np.complex128)
    M_expect = np.zeros(len(beta), dtype=np.complex128)
    
    M_expect[:] = np.einsum('bi,ij,ji->b', p, pIp, pBp)
    
    for k in range(len(w)):
        
        pAp1 = np.einsum('...i,ij,ij->...ij', p, pAp , delta(w[k] + delta_E)) 
        pABp = np.einsum('...ij,ji->...', pAp1, pBp)

        results_T0[k] = np.einsum('i,i->', p0Ap, (pBp0 * delta(w[k] - E1)))
        results_FT[k, :] = pABp
            
        print("Peak memory used for exact solution w={0:.1f}:{1:.4f} MB".format(w[k], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    
    return E1, V1, results_FT, Z, results_T0, M_expect

def dsf_lanczos(A, B, H, E0, k_d, N_samples, w, beta):

    ### lanczos method
    
    A_M = A.tocsr()
    B_M = B.tocsr()
    
    dsfT_list = []
    dsfZ_list = []
    # M_list = []
    
    # calculate iterations
    for ni in range(N_samples):
        
        ti = time()
        
        # generate normalized random vector
        r1 = np.random.normal(0, 1, size=H.Ns)
        r1 /= np.linalg.norm(r1)
        
        Br = B.dot(r1)
        r2 = Br/np.linalg.norm(Br)
        
        # get lanczos basis
        E1, V1, lv1 = lanczos_full(H, r1, k_d, eps=1e-8, full_ortho=True)
        E2, V2, lv2 = lanczos_full(H, r2, k_d, eps=1e-8, full_ortho=True)
        
        print(f'Time to finish lanczos basis {ni+1}: {time()-ti} sec')
        ti = time()
        
        psi_i = np.einsum('ij,ik->jk', V1, lv1) # N_L * N_total
        psi_j = np.einsum('ij,ik->jk', V2, lv2)
        
        # psi_0 = psi_i[0, :]
        
        # shift energy to avoid overflows
        E1 -= E0
        E2 -= E0
        delta_E = E1[:, None]-E2[None, :] 
        # print(E1)
        
        # r_psi_i = r1 @ psi_i.T
        r_psi_i = np.einsum("i,ji", r1, psi_i)
        # r_psi_i = r1.T.conj() @ psi_0 # <r|psi_i(E)>, len=NO
        psi_iApsi_j = psi_i.conj() @ A_M @ psi_j.T
        # psi_iApsi_j = np.einsum("ij,jk,kl->il", psi_i.conj(), A_M, psi_j.T)
        # psi_jB_r = psi_j.conj() @ B_M @ r1
        psi_jB_r = np.einsum("ij,j", psi_j.conj(), Br)
        # psi_j_r = np.einsum("ij,j", psi_j.conj(), r1)
        
        results_FT = np.zeros((len(w), len(beta)), dtype=np.complex128)
        # expect = np.zeros(len(beta), dtype=np.complex128)
        
        exp_factor = np.exp(-np.outer(beta, E1))
        r_psi_T = exp_factor * r_psi_i
            
        # expect[:] = np.einsum('bi,ij,j->b', r_psi_T, psi_iApsi_j, psi_j_r)
        
        for k in range(len(w)):
            
            delta_w = delta(w[k] + delta_E)
            pAp = psi_iApsi_j * delta_w

            results_FT[k, :] = np.einsum('bi,ij,j->b', r_psi_T, pAp, psi_jB_r)
        
        # compute Id
        c = np.einsum("j,aj,...j->a...", V1[0, :], V1, exp_factor)
        Id_T = np.squeeze(c[0, ...])
        
        # save results to a list
        dsfT_list.append(results_FT)
        dsfZ_list.append(Id_T)
        # M_list.append(expect)
            
        
        del E1, V1, lv1, E2, V2, lv2, results_FT, Id_T
        gc.collect()
        
    return dsfT_list, dsfZ_list# , M_list

def dsf_lanczos_T0(A, B, H, E0, k_d, N_samples, w):

    ### lanczos method
    
    A_M = A.tocsr()
    B_M = B.tocsr()
    
    dsfT_list = []
    
    # calculate iterations
    for ni in range(N_samples):
        
        ti = time()
        
        # generate normalized random vector
        r1 = np.random.normal(0, 1, size=H.Ns)
        r1 /= np.linalg.norm(r1)
        
        Br = B.dot(r1)
        r2 = Br/np.linalg.norm(Br)
        
        # get lanczos basis
        E1, V1, lv1 = lanczos_full(H, r1, k_d, eps=1e-8, full_ortho=True)
        E2, V2, lv2 = lanczos_full(H, r2, k_d, eps=1e-8, full_ortho=True)
        
        print(f'Time to finish lanczos basis {ni+1}: {time()-ti} sec')
        ti = time()
        
        psi_i = np.einsum('ij,ik->jk', V1, lv1)
        psi_j = np.einsum('ij,ik->jk', V2, lv2)
        
        psi_0 = psi_i[0, :]
        
        # shift energy to avoid overflows
        # E1 -= E0
        # print(E1[0])
        E2 -= E0
        delta_E = -E2[None, :] 
        # print(E2)
        
        r_psi_i = r1 @ psi_0.T
        # r_psi_i = r1.T.conj() @ psi_0 # <r|psi_i(E)>, len=NO
        psi_iApsi_j = psi_0.conj() @ A_M @ psi_j.T
        # psi_iApsi_j = [(psi_0.conj() @ A_M[i] @ psi_j.T[:, :, i]) for i in range(NO)]
        psi_jB_r = psi_j.conj() @ B_M @ r1
        
        results_FT = np.zeros(len(w), dtype=np.complex128)
        
        # exp_factor = np.exp(-np.outer(beta, E1))
        # r_psi_T = exp_factor * r_psi_i
            
        for k in range(len(w)):
            
            delta_w = delta(w[k] + delta_E)
            pAp = psi_iApsi_j * delta_w
            
            pAppBr = pAp @ psi_jB_r

            results_FT[k] = r_psi_i * pAppBr[0]
        
        # compute Id
        # c = np.einsum("j,aj,...j->a...", V1[0, :], V1, exp_factor)
        # Id_T = np.squeeze(c[0, ...])
        
        # save results to a list
        dsfT_list.append(results_FT)
        # dsfZ_list.append(Id_T)
            
        
        del E1, V1, lv1, E2, V2, lv2, results_FT
        gc.collect()
        
    return dsfT_list

def dsf_lanczos_src(A, B, H, E0, psi_0, k_d, N_samples, w, beta=0):

    ### lanczos method
    
    A_M = []
    B_M = []
    NO = len(A)
    
    # expect a list of A, B
    for i in range(NO):
        A_M.append(A[i].tocsr())
        B_M.append(B[i].tocsr())
    
    dsfT_list = []
    dsfZ_list = []
    
    # calculate iterations
    for ni in range(N_samples):
        
        ti = time()
        
        # generate normalized random vector
        r1 = np.random.normal(0, 1, size=H.Ns)
        r1 /= np.linalg.norm(r1)
        
        Br = [B[i].dot(r1) for i in range(NO)]
        r2 = [Br[i]/np.linalg.norm(Br[i]) for i in range(NO)]
        
        # get lanczos basis
        E1, V1, lv1 = lanczos_full(H, r1, k_d, eps=1e-8, full_ortho=True)
        # E2, V2, lv2 = lanczos_full(H, r2, k_d, eps=1e-8, full_ortho=True)
        
        results = [lanczos_full(H, r2[i], k_d, eps=1e-8, full_ortho=True) for i in range(NO)]
        E2, V2, lv2 = zip(*results)
        V2 = np.stack(V2)
        lv2 = np.stack(lv2)
        
        print(f'Time to finish lanczos basis {ni+1}: {time()-ti} sec')
        ti = time()
        
        psi_i = np.einsum('ij,ik->jk', V1, lv1)
        psi_j = np.einsum('aij,aik->ajk', V2, lv2)
        print(f"psi_0 @ psi_i: {psi_0.conj() @ psi_i[0, :]}")
        # print(f"einsum psi_0 psi_i: {np.einsum('i,i', psi_0.conj(), psi_i[0, :])}")
        
        # shift energy to avoid overflows
        E1 -= E0
        E2 -= E0
        delta_E = [-E2[i, None, :] for i in range(NO)]
        
        r_psi_i = r1.T.conj() @ psi_i.T # <r|psi_i(E)>, len=NO
        # r_psi_i = r1.T.conj() @ psi_0 # <r|psi_i(E)>, len=NO
        psi_iApsi_j = [(psi_i.conj() @ A_M[i] @ psi_j.T[:, :, i]) for i in range(NO)]
        # psi_iApsi_j = [(psi_0.conj() @ A_M[i] @ psi_j.T[:, :, i]) for i in range(NO)]
        psi_jB_r = np.einsum('...ij,...j->...i', psi_j.conj(), Br)
        
        results_FT = np.zeros((len(w), len(beta), NO), dtype=np.complex128)
        
        exp_factor = np.exp(-np.outer(beta, E1))
        r_psi_T = exp_factor * r_psi_i
            
        for k in range(len(w)):
            
            delta_w = delta(w[k] + delta_E)
            pAp = psi_iApsi_j * delta_w

            results_FT[k, :, :] = np.einsum('bi,...ij,...j->b...', r_psi_T, pAp, psi_jB_r)
        
        # compute Id
        c = np.einsum("j,aj,...j->a...", V1[0, :], V1, exp_factor)
        Id_T = np.squeeze(c[0, ...])
        
        # save results to a list
        dsfT_list.append(results_FT)
        dsfZ_list.append(Id_T)
            
        
        del E1, V1, lv1, E2, V2, lv2, results_FT, Id_T
        gc.collect()
        
    return dsfT_list, dsfZ_list