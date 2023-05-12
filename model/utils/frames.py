import torch
import torch.nn
import numpy as np

from .pdb_parser import get_xyz, select_from_mol


def get_frames(pdb, missing=None):
    """ Function for calculating "frames" for N-CA-C atoms of a list of residues
        Calibrated against 'rigids_from_3_points' in AF2 r3.py (Jumper et al., 2021, Nature)
    Args:
        pdb: pdb in a numpy structured array format
    Returns:
        (R,t): R - (nres,3,3) rotation matrix
               t - (nres,3) translation matrix
    """

    if missing and len(missing) != 0:
        missing_pad = []
        for m in missing:
            for n in ['N', 'CA', 'C']:
                missing_pad.append(
                    ('ATOM', 0, n, '', 'PAD', 'A',
                     m, 0., 0., 0., 0., pdb[0]['b'][0])
                )

        missing_pad = np.array(
            missing_pad,
            dtype=[('type', 'U6'), ('i', 'i4'), ('n', 'U4'), ('alt', 'U1'),
                   ('resn', 'U3'), ('chain', 'U2'), ('resi', 'i4'), ('x', 'f8'),
                   ('y', 'f8'), ('z', 'f8'), ('occ', 'f8'), ('b', 'f8')]
        )

        pdb = [np.sort(np.concatenate((pdb[0], missing_pad)), order='resi')]

    x1 = get_xyz(select_from_mol(pdb, 'n', ['N'])[0], gaps=False)
    x2 = get_xyz(select_from_mol(pdb, 'n', ['CA'])[0], gaps=False)
    x3 = get_xyz(select_from_mol(pdb, 'n', ['C'])[0], gaps=False)
    R, t = rigidFrom3Points(x1, x2, x3, epsilon=1e-8)

    return (torch.tensor(R), torch.tensor(t))


def rigidFrom3Points(x1, x2, x3, epsilon=1e-8):
    """ Re-written from Jumper et al., 2021 Suppl. Alg. 21 'rigidFrom3Points'
        Calibrated against 'rigids_from_3_points' in r3.py

    [Vectorised version]

    Creates a set of rigid transformations from 3 points using the Gram-Schmidt
    orthonormalisation procedure.
    Args:
        x1: [x,y,z] of N atom
        x2: [x,y,z] of CA atom
        x3: [x,y,z] of C atom
        epsilon: added to normalisation, 1e-8 in paper
    Returns:
        (R,t) where R is the (nres,3,3) rotation matrix and t is the (nres,3) translation matrix
        that compose the rigid transforms needed to transform the original coordinates
        to the basis vectors.

        R consists of [e1,e2,e3] where e1,e2,e3 are 1x3 basis vectors.
        e1 and e2 are in the CA-N and CA-C axis respectively
        e3 is the cross-product of e1 and e2.
    """

    v1 = x2-x1 #x3-x2  # in Supp. Alg. 21: v1=x3-x2 / in r3.py: v1=x2-x1
    v2 = x3-x2 #x1-x2  # in Supp. Alg. 21: v2=x1-x2 / in r3.py: v2=x3-x2
    v1_norm = np.sqrt(
        np.square(v1[0]) + np.square(v1[1]) + np.square(v1[2]) + epsilon)

    e1 = v1/v1_norm  # 1st basis vector

    u2 = v2 - (e1*np.sum(e1*v2, axis=0))
    u2_norm = np.sqrt(
        np.square(u2[0]) + np.square(u2[1]) + np.square(u2[2]) + epsilon)
    e2 = u2/u2_norm  # 2nd basis vector

    e3 = np.cross(e1.T, e2.T)  # 3rd basis vector

    R = np.concatenate(
        (e1.reshape(-1, 3, 1), e2.reshape(-1, 3, 1), e3.reshape(-1, 3, 1)), axis=2)

    return (R, x2.T)
