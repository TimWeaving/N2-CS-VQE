from pyscf import gto, scf, cc, ci, mp, fci, lib, mcscf
from pyscf.cc.addons import spatial2spin
from pyscf.scf.addons import get_ghf_orbspin
from pyscf.mcscf.addons import make_natural_orbitals
import numpy as np
import scipy as sp

def get_single_amplitudes(t1):
    no, nv = t1.shape
    nmo = no + nv
    occ_mask = np.zeros(nmo, dtype=bool)
    occ_mask[:no] = True

    indices = np.arange(0,nmo)
    single_mask = np.ix_(indices[~occ_mask], indices[occ_mask])
    double_mask = np.ix_(indices[~occ_mask], indices[occ_mask], 
                            indices[~occ_mask], indices[occ_mask])

    # dictionary of single aplitudes of form {(i,j):t_ij}
    single_amplitudes = np.zeros((nmo, nmo))
    single_amplitudes[single_mask] = t1.T
    return single_amplitudes

def entropy(noons):
    return - np.sum( np.log(noons/2) * noons/2 )

seps = np.linspace(0.8,2,160)
data_out = {
    'bond_lengths':seps.tolist(),
    'MO':[],
    'T1':[],
    'D1':[],
    'entropies':{
        'MP2':[],'CISD':[],'CCSD':[],'FCI':[],
        'CASCI(4,2)':[],'CASCI(5,4)':[],'CASCI(6,6)':[],'CASCI(7,8)':[],
        'CASSCF(4,2)':[],'CASSCF(5,4)':[],'CASSCF(6,6)':[],'CASSCF(7,8)':[]
    },
    'energies':{
        'HF':[],'MP2':[],'CISD':[],'CCSD':[],'CCSD(T)':[],'FCI':[],
        'CASCI(4,2)':[],'CASCI(5,4)':[],'CASCI(6,6)':[],'CASCI(7,8)':[],
        'CASSCF(4,2)':[],'CASSCF(5,4)':[],'CASSCF(6,6)':[],'CASSCF(7,8)':[]
    }
}

for r in seps:
    print(f'*************** r={r:.3f} ***************')
    mol = gto.Mole(atom=[('N',(0,0,0)),('N',(0,0,r))]).build()
    ### ROHF ###
    rhf = scf.ROHF(mol)
    rhf.verbose=0
    rhf.kernel()
    data_out['MO'].append(rhf.mo_energy.tolist())
    data_out['energies']['HF'].append(rhf.e_tot)
    ### MP2 correlation entropy ###
    rhf_mp2 = mp.MP2(rhf).run()
    rhf_NOONs_mp2,natorbs = make_natural_orbitals(rhf_mp2)
    data_out['entropies']['MP2'].append(entropy(rhf_NOONs_mp2))
    data_out['energies']['MP2'].append(rhf_mp2.e_tot)
    ### CISD correlation entropy ###
    rhf_cisd = ci.CISD(rhf).run()
    rhf_NOONs_cisd,_ = make_natural_orbitals(rhf_cisd)
    data_out['entropies']['CISD'].append(entropy(rhf_NOONs_cisd))
    data_out['energies']['CISD'].append(rhf_cisd.e_tot)
    ### CCSD correlation entropy ###
    rhf_ccsd = cc.CCSD(rhf).run()
    rhf_NOONs_ccsd,_ = make_natural_orbitals(rhf_ccsd)
    data_out['entropies']['CCSD'].append(entropy(rhf_NOONs_ccsd))
    data_out['energies']['CCSD'].append(rhf_ccsd.e_tot)
    data_out['energies']['CCSD(T)'].append(rhf_ccsd.e_tot + rhf_ccsd.ccsd_t())
    ### CASCI(4,2) correlation entropy ###
    rhf_casci42 = mcscf.CASCI(rhf, 4, 2).run(natorbs)
    rhf_NOONs_casci42,_ = make_natural_orbitals(rhf_casci42)
    data_out['entropies']['CASCI(4,2)'].append(entropy(rhf_NOONs_casci42))
    data_out['energies']['CASCI(4,2)'].append(rhf_casci42.e_tot)
    ### CASSCF(4,2) correlation entropy ###
    rhf_casscf42 = mcscf.CASSCF(rhf, 4, 2).run(natorbs)
    rhf_NOONs_casscf42,_ = make_natural_orbitals(rhf_casscf42)
    data_out['entropies']['CASSCF(4,2)'].append(entropy(rhf_NOONs_casscf42)) 
    data_out['energies']['CASSCF(4,2)'].append(rhf_casscf42.e_tot)
    ### CASCI(5,4) correlation entropy ###
    rhf_casci54 = mcscf.CASCI(rhf, 5, 4).run(natorbs)
    rhf_NOONs_casci54,_ = make_natural_orbitals(rhf_casci54)
    data_out['entropies']['CASCI(5,4)'].append(entropy(rhf_NOONs_casci54))
    data_out['energies']['CASCI(5,4)'].append(rhf_casci54.e_tot)
    ### CASSCF(5,4) correlation entropy ###
    rhf_casscf54 = mcscf.CASSCF(rhf, 5, 4).run(natorbs)
    rhf_NOONs_casscf54,_ = make_natural_orbitals(rhf_casscf54)
    data_out['entropies']['CASSCF(5,4)'].append(entropy(rhf_NOONs_casscf54)) 
    data_out['energies']['CASSCF(5,4)'].append(rhf_casscf54.e_tot) 
    ### CASCI(6,6) correlation entropy ###
    rhf_casci66 = mcscf.CASCI(rhf, 6, 6).run(natorbs)
    rhf_NOONs_casci66,_ = make_natural_orbitals(rhf_casci66)
    data_out['entropies']['CASCI(6,6)'].append(entropy(rhf_NOONs_casci66))
    data_out['energies']['CASCI(6,6)'].append(rhf_casci66.e_tot)
    ### CASSCF(6,6) correlation entropy ###
    rhf_casscf66 = mcscf.CASSCF(rhf, 6, 6).run(natorbs)
    rhf_NOONs_casscf66,_ = make_natural_orbitals(rhf_casscf66)
    data_out['entropies']['CASSCF(6,6)'].append(entropy(rhf_NOONs_casscf66)) 
    data_out['energies']['CASSCF(6,6)'].append(rhf_casscf66.e_tot)
    ### CASCI(7,8) correlation entropy ###
    rhf_casci78 = mcscf.CASCI(rhf, 7, 8).run(natorbs)
    rhf_NOONs_casci78,_ = make_natural_orbitals(rhf_casci78)
    data_out['entropies']['CASCI(7,8)'].append(entropy(rhf_NOONs_casci78))
    data_out['energies']['CASCI(7,8)'].append(rhf_casci78.e_tot)
    ### CASSCF(7,8) correlation entropy ###
    rhf_casscf78 = mcscf.CASSCF(rhf, 7, 8).run(natorbs)
    rhf_NOONs_casscf78,_ = make_natural_orbitals(rhf_casscf78)
    data_out['entropies']['CASSCF(7,8)'].append(entropy(rhf_NOONs_casscf78))
    data_out['energies']['CASSCF(7,8)'].append(rhf_casscf78.e_tot)
    ### FCI correlation entropy ###
    overlap_1e = mol.intor_symmetric('int1e_ovlp')
    rhf_fci = fci.FCI(rhf).run()
    rdm1 = rhf_fci.make_rdm1(rhf_fci.ci, mol.nao, mol.nelec)
    rdm1_ao = lib.einsum('pi,ij,qj->pq', rhf.mo_coeff, rdm1, rhf.mo_coeff.conj())
    w, v = sp.linalg.eigh(overlap_1e @ rdm1_ao @ overlap_1e, b=overlap_1e)
    rhf_NOONs_fci = np.flip(w)
    data_out['entropies']['FCI'].append(entropy(rhf_NOONs_fci))
    data_out['energies']['FCI'].append(rhf_fci.e_tot)
    ### T1 and D1 diagnostics ###
    rhf_orbspin = get_ghf_orbspin(rhf.mo_energy,rhf.mo_occ,is_rhf=True)
    rhf_t1 = spatial2spin(rhf_ccsd.t1, orbspin=rhf_orbspin)
    rhf_t1 = get_single_amplitudes(rhf_t1)
    data_out['T1'].append(np.linalg.norm(rhf_t1)/np.sqrt(mol.nelectron))
    data_out['D1'].append(np.max(np.linalg.svd(rhf_t1)[1]))

with open('data/N2_dissociation_results.json', 'w') as outfile:
    import json
    json.dump(data_out,outfile)