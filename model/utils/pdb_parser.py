import numpy as np
from collections import Counter
from .build_info import SPECIAL_AA, EXCLUDE_AA

def write_pdb(mol, name, comments=None):
    """ Writes a pdb file to disk """
    with open(name, 'w') as f:
        # f.write('# {}\n' % (name))
        for line in mol:
            f.write("ATOM  %5d  %-4s%s %-1s%4d    %8.3f%8.3f%8.3f%6.2f%6d\n" % (
                line['i'], line['n'], line['resn'], line['chain'], line['resi'],
                line['x'], line['y'], line['z'], line['occ'], line['b']))

        f.write('END\n\n')

        if comments:
            for line in comments:
                f.write("REMARK  %s\n" % line)


def open_pdb(file: str, pdb_chain: str="A", ignore_H: bool = False):
    """ Return the coordinates of a PDB file in a numpy structured array.
        Residues declared in 'exclude' will be ignored. Declared in 'specials'
        will be read even if the residue is a HETATM.
        Only the first model is read (up to the first ENDMDL record).

        Inputs:
            file        str         path to pdb file

        Outputs:
            molecules   np.ndarray  structured array containing 12 fields.
                                    see dtype declared below.
    """

    exclude = EXCLUDE_AA
    specials = list(SPECIAL_AA.keys())

    only_atoms = ['C', 'N', 'CA', 'O']

    dtype = [
        ('type', 'U6'), ('i', 'i4'), ('n', 'U4'), ('alt', 'U1'),
        ('resn', 'U3'), ('chain', 'U2'), ('resi', 'i4'), ('x', 'f8'),
        ('y', 'f8'), ('z', 'f8'), ('occ', 'f8'), ('b', 'f8'), ('conf', 'f8')
    ]

    molecules = []
    with open(file, 'r') as enspdbfile:
        mol = []
        pdbfile = enspdbfile.readlines()
        
        for i, line in enumerate(pdbfile):
            if line[:4] == 'ATOM' or (line[:6] == 'HETATM' and line[17:20].strip() in specials):
                if not np.char.isnumeric(line[12]) and line[17:20].strip() not in exclude:
                    n = line[12:16].strip()
                    resn = line[17:20].strip()

                    if resn in specials:
                        resn = SPECIAL_AA[resn]['convert-to']

                    if ignore_H and n[0] == 'H':
                        continue

                    if n in only_atoms:
                        if line[20:22].strip() == pdb_chain:
                            mol.append(
                                (
                                    line[:6].strip(),       # type, e.g. ATOM
                                    line[6:11],             # i, atom number
                                    n,                      # n, atom name
                                    line[16:17].strip(),    # alt, alternative id
                                    resn,                   # resn, residue name
                                    line[20:22].strip(),    # chain
                                    line[22:26],            # resi, residue id
                                    line[30:38],            # x coordinate
                                    line[38:46],            # y coordinate
                                    line[46:54],            # z coordinate
                                    line[54:60],            # occupancy
                                    line[60:66],            # b-factor
                                    0.                      # confidence 
                                )
                            )

            if line[:6] == 'ENDMDL' or line[:3] == 'END' or i == len(pdbfile)-1: # or line[:3] == 'TER':
                mol = np.array(mol, dtype=dtype)
                molecules.append(mol)
                mol = []
                break
    if len(molecules[0]) == 0:
        print("Chain ID given not present in PDB file")
        exit(128)   
    return molecules


def check_special_aa(res: np.ndarray):
    """ Simply maps some non-standard residues to standard ones e.g. MSE -> MET

        Inputs:
            res         np.ndarray      residue structured array.

        Outputs:
            res         np.ndarray      If residue is not listed in mapping,
                                        returns the original res array.
                                        Otherwise converts 'resn' and 'n' of
                                        atom arrays in the res object.
    """

    resn = res['resn'][0]
    if resn in SPECIAL_AA.keys():
        res['type'] = 'ATOM'
        res['resn'] = SPECIAL_AA[resn]['convert-to']

        for n in SPECIAL_AA[resn]['atom-mapping'].keys():
            tmp_res = res[res['n'] == n]
            tmp_res['n'] = SPECIAL_AA[resn]['atom-mapping'][n]
            res[res['n'] == n] = tmp_res

        to_remove = SPECIAL_AA[resn]['remove']
        if len(to_remove) > 0:
            res = np.array([r for r in res if r['n'] not in to_remove])

    return res


def special_aa_mapping(mol: np.ndarray):
    """ Checks for residues listed in SPECIAL_AA. Maps special to standard
        residues.

        Inputs:
            mol     np.ndarray      Molecule-level array. Loops over any
                                    special residues designated in SPECIAL_AA.

        Outputs:
            mol     np.ndarray      Returns mol array with amended residues
                                    using check_special_aa().
                                    Specifically alters atom 'type', 'resn', 'n'
                                    Returns original mol if no specials found.
    """

    spc_res = mol[np.isin(mol['resn'], list(SPECIAL_AA.keys()))]

    if len(spc_res) != 0:
        spc_resi = list(set(spc_res['resi']))

        new_residues = []
        for id in spc_resi:
            res = check_special_aa(spc_res[spc_res['resi'] == id])
            new_residues.append(res)

        mol = np.concatenate(
            (mol[~np.isin(mol['resn'], list(SPECIAL_AA.keys()))],
             np.concatenate(new_residues))
        )

        sort_id = np.argsort(mol, order=['i'])
        mol = mol[sort_id]

    return mol


def select_from_mol(mols, name, selection_list):
    """ Extract rows of mol with 'name' in 'selection_list'
        e.g. ['ASN','ASP'] from 'resn'
        e.g. ['CA'] from 'n'
        e.g. [2, 3, 4, 5] from 'resi'
        e.g. ['A','B'] from 'chain' """

    molecule = []
    for m in mols:
        molecule.append(m[np.isin(m[name], selection_list)])

    return molecule


def get_xyz(mol, gaps=True):
    """ Neater way to return xyz coordinates from mol """

    if gaps:

        missing = []
        start_res = mol[0]['resi']
        end_res = mol[-1]['resi']
        expected_range = range(start_res, end_res+1)

        mol_padded = []
        for p in expected_range:
            res = mol[mol['resi'] == p]
            if len(res) == 0:
                mol_padded.append((p, 'PAD', np.NAN, np.NAN, np.NAN))
                missing.append(p)
            else:
                mol_padded.append(
                    (res['resi'], res['resn'][0], res['x'], res['y'], res['z']))

        mol = np.array(mol_padded, dtype=[('resi', 'i4'),
                                          ('resn', 'U3'),
                                          ('x', 'f8'),
                                          ('y', 'f8'),
                                          ('z', 'f8')])

        resi = np.array(mol['resi'])
        resn = np.array(mol['resn'])
        coords = np.array([mol['x'], mol['y'], mol['z']])
        missing_resi = missing

        return resi, resn, coords, missing_resi
    else:
        return np.array([mol['x'], mol['y'], mol['z']])


def check_bb(mol, missing=False):
    """ Checks each residue for N-CA-C atoms.
        Removes any residues with missing atoms
        If missing == True, then also return any residues
        with missing backbone atoms.
    """
    missing_res = []
    new_model = []

    for m in mol:
        complete_bb = []
        missing_bb = []

        resi = set(m['resi'])
        for r in resi:
            res = m[m['resi'] == r]
            if res[res['n'] == 'N']:
                if res[res['n'] == 'CA']:
                    if res[res['n'] == 'C']:
                        if res[res['n'] == 'O']:
                            complete_bb.append(res)
                        else:
                            missing_bb.append(res)
                    else:
                        missing_bb.append(res)
                else:
                    missing_bb.append(res)
            else:
                missing_bb.append(res)

        new_model.append(np.concatenate(complete_bb))

        if len(missing_bb) != 0:
            missing_res.append(np.concatenate(missing_bb))

    if missing:
        return new_model, missing_res
    else:
        return new_model


def check_alt_res(mol):
    """ Checks for alternative residue locations using
    occupancy. If multiple sets of atoms present for a
    residue, selects set with highest occupancy.
    """

    new_model = []

    for m in mol:
        single_occ_model = []

        resi = set(m['resi'])
        for r in resi:

            res = m[m['resi'] == r]
            counts = Counter(res['n'])

            if counts.most_common(1)[0][1] > 1:
                for ele in counts:
                    res_sort = np.sort(res[res['n'] == ele], order='occ')
                    single_occ_model.append([res_sort[-1]])
            else:
                single_occ_model.append(res)

        new_model.append(np.concatenate(single_occ_model))

    return new_model
