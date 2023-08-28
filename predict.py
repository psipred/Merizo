from __future__ import annotations

import os
import time
import argparse
import textwrap

import torch

from model.network import Merizo

from model.utils.features import generate_features_domain
from model.utils.utils import (
    get_device,
    format_dom_str,
    instance_matrix,
    write_pdf_predictions,
    write_pdb_predictions,
    write_domain_idx,
    write_fasta,
    clean_domains,
    clean_singletons,
    get_ids,
    remap_ids,
    shuffle_ids,
    separate_components,
)

# --- Constants for cleanup and iterative segmentation

MIN_DOMAIN_SIZE = 50    # minimum number of residues in a domain
MIN_FRAGMENT_SIZE = 10  # minimum number of residues in a single segment
DOM_AVE = 200           # half of the average domain size of CATH / for iteration mode
CONF_THRESHOLD = 0.75   # minimum domain confidence / for iteration mode

def iterative_segmentation(
        network: torch.nn.Module, 
        inputs: tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor], 
        domain_ids: torch.tensor, 
        conf_res: torch.tensor,
        max_iterations: int,
    ) -> tuple[torch.tensor, torch.tensor]:
    """_summary_

    Args:
        network (torch.nn.Module): _description_
        inputs (tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]): _description_
        domain_ids (torch.tensor): _description_
        conf_res (torch.tensor): _description_

    Returns:
        tuple[torch.tensor, torch.tensor]: _description_
    """
    
    n_iterations = 0
    iterate = True
    ignore_index = []
    
    while iterate:
        ids, _ = get_ids(domain_ids)
        unique_ids = {}
        
        for d in ids:
            if d.item() not in ignore_index:
                dom_mask = domain_ids == d
                dom_conf_res = conf_res[dom_mask]
                dom_nres = len(dom_conf_res)
                
                assert len(dom_conf_res.unique()) == 1
                
                dom_conf = dom_conf_res.mean()
                
                cond1 = dom_nres > DOM_AVE
                # cond2 = dom_conf < CONF_THRESHOLD
                
                if cond1: # or cond2:
                    unique_ids[d.item()] = dom_conf
                else:
                    ignore_index.append(d.item())
                    
        if len(unique_ids) == 0:
            iterate = False
            break
        
        counter = 1
        for k, dconf in unique_ids.items():
            domain_mask = domain_ids == k
            
            domain_ids_, conf_res_ = network(inputs, mask=domain_mask)
            
            ids, ndoms_ = get_ids(domain_ids_)

            # Get new confidence scores per new domain
            new_dom_conf = torch.zeros_like(ids).float()
            for i, d in enumerate(ids):
                new_mask = domain_ids_ == d
                new_dom_conf[i] = conf_res_[new_mask].mean()
                
            # If the segment is still 1 domain, skip
            if ndoms_ == 1:
                ignore_index.append(k)
                
            # Otherwise split and overwrite old domain ids and confidences
            else:
                # Offset new ids by at least no_classes to ensure no overlap
                # with the old ids, and then transplant back in
                dd = domain_ids_ + (counter * network.no_classes)
                dd[domain_ids_ == 0] = 0
                domain_ids[domain_mask] = dd

                # Assign all residues in the new domain with the predicted
                # domain confidence
                conf_res[domain_mask] = conf_res_
                counter += 1
                
        n_iterations += 1
        if n_iterations == max_iterations:
            iterate = False
            break
        
    return domain_ids, conf_res

def read_split_weight_files(directory):
    weights = {}
    
    # Get the list of weight files in the directory
    weight_files = [file for file in os.listdir(directory) if file.endswith('.pt')]
    
    # Read the weights from each file
    for file in weight_files:
        file_path = os.path.join(directory, file)
        subset_weights = torch.load(file_path)
        
        # Add the subset weights to the overall weights dictionary
        weights.update(subset_weights)
    
    return weights


def run_merizo():
    # Read the config file
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            If you use Merizo, please cite the following paper:
                Lau, et al., 2023. Merizo: a rapid and accurate domain segmentation method using invariant point attention. bioRxiv, doi: https://doi.org/10.1101/2023.02.19.529114
            
            Example usage:
                python predict.py -d cpu -i examples/2xdqA.pdb
                python predict.py -d cpu -i examples/*.pdb --save_domains --save_pdf --save_fasta
                python predict.py -d cpu -i examples/2xdqA.pdb --save_domains --plddt_filter
                
            For AlphaFold2 models, the iterative segmentation routine may give better results on longer models:
                python predict.py -d cpu -i examples/AF-Q96PD2-F1-model_v4.pdb --iterate --plddt_filter 60 --conf_filter 0.75
         ''')
    )
    
    parser.add_argument("-i", "--input", type=str, nargs="+", required=True, help="Specify path to pdb file input. Can also take multiple inputs (e.g. '/path/to/file.pdb' or '/path/to/*.pdb').")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Hardware to run on. Options: 'cpu', 'cuda', 'mps'.")
    parser.add_argument("--save_pdf", dest="save_pdf", action="store_true", help="Include to save the domain map as a pdf.")
    parser.add_argument("--save_pdb", dest="save_pdb", action="store_true", help="Include to save the result as a pdb file. All domains will be included unless --conf_filter or --plddt_filter is used.")
    parser.add_argument("--save_domains", dest="save_domains", action="store_true", help="Include to save parsed domains as separate pdb files. Also saves the full pdb.")
    parser.add_argument("--save_fasta", dest="save_fasta", action="store_true", help="Include to save a fasta file of the input pdb.")
    parser.add_argument("--conf_filter", dest="conf_filter", type=float, default=None, help="(float, [0-1]) If specified, only domains with a pIoU above this threshold will be saved. ")
    parser.add_argument("--plddt_filter", dest="plddt_filter", type=float, default=None, help="(float, [0-1]) If specified, only domain with a plDDT above this threshold will be saved. Note: if used on a non-AF structure, this will correspond to crystallographic b-factors.")
    parser.add_argument("--iterate", dest="iterate", action="store_true", help=f"If used, domains under a length threshold (default: {DOM_AVE} residues) will be re-segmented.")
    parser.add_argument("--max_iterations", dest="max_iterations", type=int, default=3, help="(int [1, inf]) Specify the maximum number of re-segmentations that can occur.")
    parser.add_argument("--shuffle_indices", dest="shuffle_indices", action="store_true", help="Shuffle domain indices - increases contrast between domain colours in PyMOL.")
    parser.add_argument("--return_indices", dest="return_indices", action="store_true", help="Return the domain indices for all residues.")
    
    args = parser.parse_args()

    label = "merizo_v2"

    if args.iterate:
        label = label + f'-iterate-{args.max_iterations}'

    device = get_device(args.device)
    network = Merizo().to(device)

    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    network.load_state_dict(read_split_weight_files(weights_dir), strict=True)
    network.eval()

    with torch.no_grad():
        for pdb_path in args.input:
            if os.path.exists(pdb_path):
                start_time = time.time()

                pdb_name = os.path.basename(pdb_path)
                fn, _ = os.path.splitext(pdb_path)
                outname = fn + "_" + label
                
                if args.return_indices:
                    dom_idx_name = outname + '.idx'

                if not os.path.exists(outname):
                    
                    try: 
                        s, z, r, t, ri, pdb, _ = generate_features_domain(pdb_path, device)
                        nres = s.shape[1]

                        inputs = (s, z, r, t, ri)
                        domain_ids, conf_res = network(inputs)

                        # If --iterate mode, iteratively segment domains 
                        if args.iterate:
                            if nres > DOM_AVE * 2:
                                domain_ids, conf_res = iterative_segmentation(
                                    network, inputs, domain_ids, conf_res, args.max_iterations)

                        R_pred = instance_matrix(domain_ids)[0]
                        
                        domain_ids = separate_components(R_pred, z, domain_ids)

                        if len(torch.unique(domain_ids)) > 1:
                            domain_ids = clean_domains(domain_ids, MIN_DOMAIN_SIZE)
                            domain_ids = clean_singletons(domain_ids, MIN_FRAGMENT_SIZE)

                        # Recompute the domain map given the new assignment
                        R_pred = instance_matrix(domain_ids)[0]
                        
                        if args.shuffle_indices:
                            domain_ids = shuffle_ids(domain_ids)
                        else:
                            domain_ids = remap_ids(domain_ids)
                        
                        conf_global = conf_res.mean()
                        
                        _, ndoms = get_ids(domain_ids)

                        # --------------
                        # Outputs 
                        # --------------

                        dom_str = format_dom_str(domain_ids, ri)
                        
                        nres = domain_ids.shape[0]
                        nres_domain = domain_ids.count_nonzero()
                        nres_ndomain = nres - nres_domain

                        if args.save_pdb or args.save_domains:
                            write_pdb_predictions(
                                pdb,
                                domain_ids,
                                conf_res,
                                ri,
                                args.save_domains,
                                args.conf_filter,
                                args.plddt_filter,
                                outname,
                            )

                        if args.save_fasta:
                            write_fasta(pdb, outname, pdb_name[:-4])
                            
                        if args.return_indices:
                            write_domain_idx(dom_idx_name, domain_ids, ri)

                        if args.save_pdf:
                            R_pred = instance_matrix(domain_ids)[0]
                            p_conf = torch.sqrt(conf_res[None, :] * conf_res[:, None])
                            p_conf = p_conf * R_pred

                            title = "{} | {} predicted domains".format(pdb_name, ndoms)
                            write_pdf_predictions(R_pred, p_conf, title, outname)

                        # if args.save_domains:
                        end_time = time.time() - start_time
                        print("{}\t{}\t{}\t{}\t{}\t{:.5f}\t{:.5f}\t{}".format(
                            pdb_name,
                            nres,
                            nres_domain,
                            nres_ndomain,
                            ndoms,
                            conf_global.item(),
                            end_time,
                            dom_str,
                        ))

                    except:
                        print("{}\tSegmentation failed.".format(pdb_name))
            else:
                print("Cannot find file at {}".format(pdb_path))


if __name__ == "__main__":
    run_merizo()
