import os
import re

import numpy as np

from scipy.spatial import distance_matrix

import torch
import torch.nn.functional as F

from model.utils.frames import get_frames
from model.utils.pdb_parser import (
    select_from_mol,
    get_xyz,
    open_pdb,
    check_bb,
    check_alt_res
)


resndict = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            'PAD': 'X', 'ASH': 'D', 'GLH': 'E', 'HID': 'H', 'HIE': 'H',
            'HIP': 'H', 'HSD': 'H', 'HSE': 'H', 'LYN': 'K',
            }

aa_trans = str.maketrans('ARNDCQEGHILKMFPSTWYVBJOUXZ-.',
                         'ABCDEFGHIJKLMNOPQRSTUUUUUUVV')

def generate_features_domain(file: str, device: torch.device, pdb_chain: str="A"):
    """Returns single, pair, rotation, translation and resi index features for a pdb.

    Args:
        file (str): path to the pdb target.
        device (torch.device): torch.device object.

    Returns:
        Tuple[
            torch.tensor,   s:  [1,n,20]        single repr features
            torch.tensor,   z:  [1,n,n,1]       pair repr features
            torch.tensor,   r:  [1,n,3,3]       Gram-Schmidt rotation matrices
            torch.tensor,   t:  [1,n,3]         Gram-Schmidt translation vectors
            torch.tensor,   ri: [1,n]           residue indices for ALiBi
            np.ndarray,     pdb                 numpy structured array containing pdb records
            np.ndarray,     b:  [n]             numpy array of b-factor values
        ]
    """

    pdb, ca, backbone = pdb_to_features(file, pdb_chain)

    seq = torch.tensor(encode_seq(ca["resn"], three_letter=True)).long()

    s = F.one_hot(seq, 20)
    z = torch.tensor(get_dmap(ca, missing=False)).unsqueeze(-1)

    ri = torch.tensor(ca["resi"])
    b = torch.tensor(ca['b'])

    backbone = select_from_mol([backbone], "resi", ca["resi"])
    r, t = get_frames(backbone)

    s = s.unsqueeze(0).float().to(device)
    z = z.unsqueeze(0).float().to(device)
    r = r.unsqueeze(0).float().to(device)
    t = t.unsqueeze(0).float().to(device)
    ri = ri.unsqueeze(0).float().to(device)

    return {'s': s, 'z': z, 'r': r, 't': t, 'ri': ri, 'pdb': pdb, 'b': b, 'nres': s.shape[1]}

def pdb_to_features(file, pdb_chain: str="A", resi=None):

    if os.path.exists(file):
        pdb = open_pdb(file, pdb_chain)
        pdb = check_bb(check_alt_res(pdb), missing=False)
        pdb = [np.sort(pdb[0], order='resi')]

        if resi is not None:
            pdb = select_from_mol(pdb, 'resi', resi)

        backbone = select_from_mol(pdb, 'n', ['N', 'CA', 'C', 'O'])[0]
        ca = select_from_mol(pdb, 'n', ['CA'])[0]

        return pdb[0], ca, backbone
    else:
        print("Cannot find model {}".format(os.path.basename(file)))

def pdb_to_fasta(pdb: np.ndarray) -> str:
    """_summary_

    Args:
        pdb (np.ndarray): _description_

    Returns:
        str: _description_
    """

    return ''.join([resndict[x] for x in pdb[pdb['n'] == 'CA']['resn']])

def cath_dom_str_to_resi(c):
    """ Retrieves the first and last residue from a residue range.
        Handles negative first or negative second ids """

    c = re.sub(r'[A-Z]', '', c).replace('(', '').replace(')', '')

    # If first number is negative, temp replace with +
    if c[0] == '-':
        c = list(c)
        c[0] = '+'
        c = ''.join(c)

    # If second number is negative, temp replace with +
    if '--' in c:
        c = c.replace('--', '-+')

    a, b = c.split("-")

    a = int(a.replace('+', '-'))
    b = int(b.replace('+', '-'))

    return a, b


def boundaries_to_res(boundaries, query=None, resi=None):
    lb_resi = []
    lb_id = []
    ranges = []
    residues = []

    for c in boundaries:
        a, b = cath_dom_str_to_resi(c)

        assert a < b, f"{query}: boundary {c} - residues {a} cannot be greater than {b}"

        if resi is not None:
            # Hack to offset residues if there is 1 missing from the start/end
            orig_a, counter = a, 1
            while a not in resi:
                counter += 1
                a += 1

                assert counter < 5, f"Cannot find residue a: {orig_a} for pdb {query} in: {resi}"

            orig_b, counter = b, 1
            while b not in resi:
                counter += 1
                b -= 1

                assert counter < 5, f"Cannot find residue b: {orig_b} for pdb {query} in: {resi}"

            lb_id.append((list(resi).index(a), list(resi).index(b)))

        lb_resi.append((a, b))
        ranges.extend([a, b])
        residues.extend(range(a, b+1))

    if resi is not None:
        min_max_id = (list(resi).index(min(ranges)),
                      list(resi).index(max(ranges)))
    else:
        min_max_id = None

    domains = {
        'domain_resi': lb_resi,
        'domain_id': lb_id,
        'min_max_resi': (min(ranges), max(ranges)),
        'min_max_id': min_max_id,
        'residues': residues,
    }

    return domains


def encode_seq(seq, three_letter=False):
    """ Encodes a single-letter protein sequence into integers """
    if three_letter:
        seq = [resndict[s] for s in seq]

    return (np.frombuffer(''.join(seq).translate(aa_trans).encode(
        'latin-1'), dtype=np.uint8) - ord('A'))  # .view(len(seq))


def get_dmap(mol, missing=True):
    """ Returns (distance matrix, resi list) for a molecule

    Inputs:
        mol: pdb in a numpy structured array
        missing: if True, returns distance map accounting for missing residues
                 if False, returns distance map without gaps
    """

    if missing:
        resi, resn, coords, missing_resi = get_xyz(mol, gaps=True)
        dm = distance_matrix(coords.T, coords.T)
        return resi, resn, coords, dm, missing_resi
    else:
        coords = get_xyz(mol, gaps=False)
        dm = distance_matrix(coords.T, coords.T)
        return dm


def get_hmap(mol):
    """ Return the backbone hydrogen bonding map """

    mol = mol[np.argsort(mol, order=['resi'])]
    n_atoms = mol[mol['n'] == 'N']
    o_atoms = mol[mol['n'] == 'O']

    ri = abs(n_atoms['resi'][..., None] - n_atoms['resi'][..., None, :])
    hmap_no = distance_matrix(
        get_xyz(n_atoms, gaps=False).T,
        get_xyz(o_atoms, gaps=False).T
    )

    hmap_no[(hmap_no >= 3.5) | (ri < 3)] = 0
    hmap_no[hmap_no != 0] = 1

    hmap_on = distance_matrix(
        get_xyz(o_atoms, gaps=False).T,
        get_xyz(n_atoms, gaps=False).T
    )

    hmap_on[(hmap_on >= 3.5) | (ri < 3)] = 0
    hmap_on[hmap_on != 0] = 1

    hmap = torch.cat((torch.tensor(hmap_no).unsqueeze(-1),
                      torch.tensor(hmap_on).unsqueeze(-1)), dim=-1)

    assert hmap.shape[0] == hmap.shape[1]

    return hmap


def cent_to_dist(
    cent, min_bin=0, max_bin=90, no_bins=10, inf=1e8
):
    lower = torch.linspace(min_bin, max_bin, no_bins)
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)

    dg = cent.unsqueeze(-1).expand(-1, -1, no_bins)

    return ((dg > lower) * (dg < upper)).type(cent.dtype)

def dm_to_distogram(dm, min_bin=3.25, max_bin=50.75, no_bins=39, inf=1e8):
    lower = torch.linspace(min_bin, max_bin, no_bins)
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)

    dg = dm.unsqueeze(-1).expand(-1, -1, -1, no_bins)
    return ((dg > lower) * (dg < upper)).float()