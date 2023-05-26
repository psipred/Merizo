from __future__ import annotations

import os

from itertools import groupby
from operator import itemgetter

from natsort import natsorted

import networkx as nx

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc

import torch

from model.utils.constants import PRIMES
from model.utils.features import pdb_to_fasta
from model.utils.pdb_parser import write_pdb, select_from_mol


def get_device(device: str) -> torch.device:
    """Returns the selected device for inference.

    Args:
        device (str): expects 'cuda' or 'cpu' from argparse.

    Returns:
        torch.device: returns selected torch.device object.
    """

    if torch.cuda.is_available() and device == "cuda":
        device = torch.device("cuda") # GPU
        
    elif device == "cpu" or (not torch.cuda.is_available() and device == "cuda"):
        device = torch.device("cpu")  # CPU
        
    elif device == "mps":
        device = torch.device("mps")  # Apple Sllicon
        
    else:
        print("Device (-d) needs to be either 'cpu', 'cuda' or 'mps'.")
        exit()

    return device


def get_ids(x: torch.tensor) -> tuple[torch.tensor, int]:
    """ Returns a tuple containing unique non-zero indices and 
        the number of unique non-zero indices.

    Args:
        tensor (torch.tensor):      [N] domain_indices

    Returns:
        tuple[torch.tensor, int]:   Tuple of unique non-zero
                                    domain indices and number of 
                                    such indices
    """

    ids = x[x.nonzero()].unique()
    return ids, len(ids)


def remap_ids(x):
    """
    Remap non-zero values in an array of integers beginning from 1 in order of occurrence.
    ** Don't use torch.unique() as the integers will be sorted
    """
    # Get the unique non-zero values in the array
    unique_vals = []
    for val in x:
        if val != 0 and val not in unique_vals:
            unique_vals.append(val.item())

    # Create a mapping from old values to new values
    mapping = {val: i + 1 for i, val in enumerate(unique_vals)}

    # Use the mapping to remap the non-zero values in the array
    remapped_arr = torch.zeros_like(x)
    for i, val in enumerate(x):
        if val != 0:
            remapped_arr[i] = mapping[val.item()]

    return remapped_arr


def shuffle_ids(domain_ids: torch.tensor) -> torch.tensor:
    """ Remap domain indices to increase contrast between
        domains in PyMOL. 

    Args:
        domain_ids (torch.tensor): [N] domain indices [0, ndoms]

    Returns:
        torch.tensor:              [N] new domain indixes [0, ndoms]
    """

    unique_values, _ = get_ids(domain_ids)
    dummy_ids = torch.randperm(len(unique_values)) + 1

    new_ids = torch.zeros_like(domain_ids)
    for u, d in zip(unique_values, dummy_ids):
        new_ids[domain_ids == u] = d

    return new_ids


def mask_by_domain(s, z, r, t, ri, domain_ids):
    """Splits inputs by predicted domain id

    Args:
        s (_type_): _description_
        z (_type_): _description_
        r (_type_): _description_
        t (_type_): _description_
        ri (_type_): _description_
        d_mask (_type_): _description_
    """

    domain_inputs = []
    domain_dict = {}
    unique_ids = domain_ids[domain_ids.nonzero()].unique()
    for u in unique_ids:
        domain_mask = domain_ids == u
        domain_inputs.append(
            (
                s[:, domain_mask],
                z[:, domain_mask][:, :, domain_mask],
                r[:, domain_mask],
                t[:, domain_mask],
                ri[:, domain_mask],
            )
        )

        domain_dict[ri[0, 0]] = 1

    return domain_inputs, domain_dict


def instance_matrix(labels):
    """Calculate the domain map given some labels.
    Note: using non-prime numbers for the label ids will lead to some
    false domains - we first need to map the labels to primes and then
    map back
    """

    labels_prime = PRIMES.to(labels.device)[labels.long()]

    R = torch.sqrt(torch.einsum("i,j->ij", labels, labels))
    R_prime = torch.sqrt(torch.einsum("i,j->ij", labels_prime, labels_prime))

    R_prime[R_prime % 1 != 0] = 0

    non_domain = R_prime.clone()
    non_domain[non_domain != PRIMES[0]] = 0
    R_prime[R_prime == PRIMES[0]] = 0

    R_clamped = torch.clamp(R_prime.clone(), min=0, max=1)
    R = R * R_clamped

    R_clamped[non_domain == PRIMES[0]] = 0  # 0.5

    return R_clamped, R  # .fill_diagonal_(1), R


def clean_domains(dom_ids: torch.tensor, min_num: int) -> torch.tensor:
    """ Divides any domains that are too small, equally into the domain before and after it.
    """

    dom_counts = torch.unique(dom_ids[dom_ids.nonzero()], return_counts=True)
    remove_ids = dom_counts[0][dom_counts[1] < min_num]

    if len(remove_ids.size()) != 0:
        for i in range(len(dom_ids)):
            if dom_ids[i].item() in remove_ids and i > 0:
                if i != 0:
                    dom_ids[i] = dom_ids[i - 1]

    return dom_ids

def assimilate_short_terminal(dom_ids, dom_counts, threshold, termini):
    """ Assimilates any short stretches of residues the N or C terminus into the 
    proceeding/preceeding domain.
    """
    if termini == 'C':
        dom_counts = dom_counts.flip([0])
        dom_ids = dom_ids.flip([0])
        
    for i, (c, d) in enumerate(zip(dom_counts, dom_ids)):
        if c >= threshold or d == 0:
            next_dom_id = dom_ids[i+1]
            break
    
    dom_ids[:i+1] = next_dom_id
    
    if termini == 'C':
        return dom_ids.flip([0])
    else:
        return dom_ids
    
def get_segment_length(dom_ids):
    """ Returns the length of each segment of a domain.
        e.g. [1,1,1,1,1,1,5,5,5,5,5,2,2,2,2,2] ->
             [6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5]
    """
    dom_counts = torch.ones_like(dom_ids)
    for i, d in enumerate(dom_ids):
        if i == 0:
            counter = 1
            _idx = idx = i
        else:
            if d == dom_ids[i - 1]:
                counter += 1
                idx = i
            else:
                dom_counts[_idx : idx + 1] = counter
                counter = 0
                _idx = i

        if i == len(dom_ids) - 1:
            dom_counts[_idx : idx + 1] = counter
            
    return dom_counts

def clean_singletons(dom_ids: torch.tensor, threshold: int) -> torch.tensor:
    """ Re-assigns short segments:
        N-terminus: assimilated into first domain
        C-terminus: assimilated into last domain
        internal segments: divided between the preceeding and proceeding domain
    """

    dom_ids_ = dom_ids
    dom_counts = get_segment_length(dom_ids)
            
    if len(dom_counts <= threshold) != 0:
        # Assimilate short N-terminal stretches
        dom_ids_ = assimilate_short_terminal(dom_ids_, dom_counts, threshold, termini='N')

        # Assimilate short C-terminal stretches
        dom_ids_ = assimilate_short_terminal(dom_ids_, dom_counts, threshold, termini='C')
        
        # Divide internal stretches of NDR that are too short
        short_ndr = (dom_counts < threshold) * (dom_ids == 0)
        non_terminal = torch.where(short_ndr.long() == 0)[0]
        
        # Trim off any N and C-terminal ndr stretches
        short_ndr[:non_terminal[0]] = False
        short_ndr[non_terminal[-1]:] = False

        # Subdivide internal NDRs between domains
        true_indices = torch.nonzero(short_ndr).flatten().tolist()

        stretches = []
        for _, g in groupby(enumerate(true_indices), lambda ix: ix[0] - ix[1]):
            stretch = [x[1] for x in g]
            stretches.append(torch.tensor(stretch))

        # Print the stretches
        for stretch in stretches:
            start, end = stretch[0].item(), stretch[-1].item()
            mid = torch.median(stretch).item()
            
            dom_ids_[start:mid+1] = dom_ids_[start-1]
            dom_ids_[mid:end+1] = dom_ids_[end+1]

    return dom_ids_


def separate_components(
        domain_map:torch.tensor, 
        z: torch.tensor, 
        domain_ids: torch.tensor, 
        distance:float=10.0
    ) -> torch.tensor:
    """ Separates domains in the domain map based on a minimum distance.
        Takes the intersect between the domain map and distance map 
        graphs to disconnect discontinuous associations between segments
        when the distance (Angstroms) between them is greater then 
        <distance>. 

    Args:
        domain_map (torch.tensor):  [N,N]   domain map [0,1]
        z (torch.tensor):           [N,N,1] distance map [0,inf]
        domain_ids (torch.tensor):  [N]     domain indices [0, ndom]
        distance (float, optional):         distance cutoff in Angstroms    

    Returns:
        torch.tensor:               [N]     updated domain indices [0, ndom]
    """

    # Re-assign indices based on distance cut off between domains
    dm = z.reshape(z.shape[1], z.shape[1])

    # Convert domain map and thresholded distance map into graphs
    # and compute intersect between both, to remove any vertices
    # in the domain map where the distance is greater than some threshold

    G_int = nx.intersection(
        nx.from_numpy_array(domain_map.detach().cpu().numpy()),
        nx.from_numpy_array(
            (dm <= distance).bool().detach().cpu().numpy()),
    )

    G_int.remove_nodes_from(list(nx.isolates(G_int)))

    # Iterate over each component and re-assign a domain index
    components = list(nx.connected_components(G_int))
    labels_new = torch.zeros_like(domain_ids)
    
    for i, c in enumerate(components):
        labels_new[list(c)] = i + 1

    return labels_new.to(z.device)


def NonLinCdict(steps, hexcol_array):
    cdict = {"red": (), "green": (), "blue": ()}
    for s, hexcol in zip(steps, hexcol_array):
        rgb = mc.hex2color(hexcol)
        cdict["red"] = cdict["red"] + ((s, rgb[0], rgb[0]),)
        cdict["green"] = cdict["green"] + ((s, rgb[1], rgb[1]),)
        cdict["blue"] = cdict["blue"] + ((s, rgb[2], rgb[2]),)

    return cdict


def write_pdf_predictions(
    R_pred: torch.tensor,
    confidence: torch.tensor,
    title: str,
    outname: str,
):
    """_summary_

    Args:
        R_pred (torch.tensor): _description_
        doms (torch.tensor): _description_
        title (torc)
        outname (str): _description_
    """

    hc = ["#e6e6f3", "#cccce6", "#9999cc", "#6666b3", "#33339a", "#000080"]
    th = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    cdict = NonLinCdict(th, hc)
    cm = mc.LinearSegmentedColormap("test", cdict)

    width = 5
    n_plots = 2  # number of static plots
    _, ax = plt.subplots(1, n_plots, figsize=(n_plots * width, width))
    plt.gcf().suptitle(title, fontsize=18)

    ax[0].matshow(R_pred.cpu().detach().numpy(), cmap=cm)
    ax[1].matshow(confidence.cpu().detach().numpy(), cmap=cm)

    ax[0].set(xlabel="Predicted Domain Map")
    ax[1].set(xlabel="Domain Confidence")

    plt.tight_layout()
    plt.savefig(outname + ".pdf", format="pdf")
    plt.close()


def write_pdb_predictions(
    pdb: np.ndarray,
    dom_ids: torch.tensor,
    conf: torch.tensor,
    ri: torch.tensor,
    save_domains: bool,
    conf_filter: bool,
    plddt_filter: bool,
    outname: str,
    comment: list = None,
):
    """_summary_
    
    Re-writes occ with domain ids, b-factors remain b-factors or AF2 plDDT. Confidence 
    estimates are in new 'conf' field.

    Args:
        pdb (np.ndarray): _description_
        dom_ids (torch.tensor): _description_
        conf (torch.tensor): _description_
        b (torch.tensor): _description_
        ri (torch.tensor): _description_
        save_domains (bool): _description_
        outname (str): _description_
        comment (list, optional): _description_. Defaults to None.
    """

    pdb["occ"] = 0

    ids, confs = [], []
    for r in pdb["resi"]:
        ids.append(dom_ids[ri.squeeze(0) == r].item())
        confs.append(conf[ri.squeeze(0) == r].item())
        
    pdb["occ"] = np.array(ids)
    pdb["conf"] = np.array(confs)
    
    name = os.path.basename(outname)

    if save_domains:
        for i, u in enumerate(dom_ids[dom_ids.nonzero()].unique()):
            output = False

            p = pdb[pdb['occ'] == u.item()]

            p_ca = select_from_mol([p], 'n', ['CA'])[0]
            dom_conf = np.mean(p_ca['conf'])
            dom_plddt = np.mean(p_ca['b'])
            
            dom_str = format_dom_str(
                torch.tensor(p_ca['occ']), 
                torch.tensor(p_ca['resi']).unsqueeze(0)
            )
            
            if conf_filter is not None and plddt_filter is None:
                if dom_conf >= conf_filter:
                    output = True
                    
            if plddt_filter is not None and conf_filter is None:
                if dom_plddt >= plddt_filter:
                    output = True
                    
            if conf_filter is not None and plddt_filter is not None:
                if dom_conf >= conf_filter and dom_plddt >= plddt_filter:
                    output = True
                    
            if conf_filter is None and plddt_filter is None:
                output = True

            if output:
                with open(outname + '.domains', 'a+') as f:
                    f.write("{}\t{:.0f}\t{}\t{:.3f}\t{:.3f}\t{:.0f}\t{}\n".format(
                        name, i+1, len(p_ca), dom_conf, dom_plddt, u.item(), dom_str
                    ))
                
                write_pdb(p, outname + "_" + str(i+1).zfill(2) + ".pdb2", comment)

    write_pdb(pdb, outname + ".pdb2", comment)


def write_fasta(pdb: np.ndarray, outname: str, header: str):
    """_summary_

    Args:
        pdb (np.ndarray): _description_
        outname (str): _description_
        header (str): _description_
    """
    fasta = pdb_to_fasta(pdb)

    with open(outname + ".fasta", "w") as f:
        f.write(">" + header + "\n")
        f.write(fasta + "\n")


def format_dom_str(dom_cons: torch.tensor, ri: torch.tensor) -> list:
    """Formats predicted domain ids into residue ranges.
        Domains delimited by ,
        Discontinuous segments delimited by _

        e.g. [0,1,2,3,4,10,11,12] -> ['0-4_10-12']

    Args:
        dom_ids (torch.tensor): predicted domain ids.
        ri (torch.tensor):      residue indices, for mapping back to
                                pdb residues.

    Returns:
        list: _description_     list of domain strings.
    """

    dom_str = []
    for d in torch.unique(dom_cons[dom_cons != 0], sorted=False):
        dom_resi = ri[0, dom_cons == d].long().cpu().tolist()

        strs = []
        for k, g in groupby(enumerate(dom_resi), lambda ix: ix[0] - ix[1]):
            consecs = list(map(itemgetter(1), g))
            if len(consecs) > 1:
                strs.append("{}-{}".format(str(consecs[0]), str(consecs[-1])))
            else:
                strs.append(str(consecs[0]))

        dom_str.append("_".join(strs))

    return ",".join(natsorted(dom_str))
