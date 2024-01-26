# Merizo

Merizo is a fast and accurate deep learning method for domain segmentation in complex protein structures. Notably, it makes use of invariant point attention (IPA) to read a protein structure into a latent representation. Domains are predicted via an affinity learning approach whereby the embeddings of residues belonging to the same domain, are encouraged towards similar embeddings and discouraged if belonging to different domains. 

If you use Merizo, please cite the paper: 

   **Lau et al., 2023. Merizo: a rapid and accurate protein domain segmentation method using invariant point attention. bioRxiv.** ([link](https://www.biorxiv.org/content/10.1101/2023.02.19.529114v2⋅⋅* ))  

## Installation

Packages needed to run Merizo can be installed either individually or concurrently using the `requirements.txt` file:
```
pip install -r requirements.txt
```

Individual dependencies:
```
pip install torch torchvision torchaudio
pip install scipy
pip install matplotlib
pip install einops
pip install networkx
pip install rotary-embedding-torch
pip install natsort
```

## Usage
The standard usage for Merizo is:
``` 
python predict.py -i /path/to/file.pdb
```

Multiple pdb files can be supplied to `-i`:
``` 
python predict.py -i /path/to/file_1.pdb /path/to/file_2.pdb /path/to/file_3.pdb
```
or:
``` 
python predict.py -i /path/to/files/*.pdb
```

Merizo can be ran either in standard mode, or an iterative segmentation mode where domains larger than a certain size are continuously passed through the network until convergence (i.e. the assignment no longer changes). This length threshold is set to 200 residues (roughly just smaller than the average domain in CATH) in `predict.py`.

The iterative mode should be used for processing any long models (~ >800 residues), as well as models from the AlphaFold Protein Structure Database.

#### Standard mode
```
python pred.py -d cpu -i examples/2xdqA.pdb
python pred.py -d cpu -i examples/*.pdb --save_domains --save_pdf --save_fasta
python pred.py -d cpu -i examples/2xdqA.pdb --save_domains --plddt_filter
```

#### Iterative mode
```
python pred.py -d cpu -i examples/AF-Q96PD2-F1-model_v4.pdb --iterate
```

By default if no device flag (`-d`) is specified, if a GPU is available, it will be used. If no GPU is available, Merizo will default to using CPU. On Apple Silicon devices, the `-d mps` option can be given to use the Metal Performance Shaders (MPS) backend.

#### Options
```
  -h, --help            show this help message and exit
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Specify path to pdb file input. Can also take multiple inputs (e.g. '/path/to/file.pdb' or '/path/to/*.pdb').
  -d DEVICE, --device DEVICE
                        Hardware to run on. Options: 'cpu', 'cuda', 'mps'.
  --save_pdf            Include to save the domain map as a pdf.
  --save_pdb            Include to save the result as a pdb file. All domains will be included unless --conf_filter or --plddt_filter is used.
  --save_domains        Include to save parsed domains as separate pdb files. Also saves the full pdb.
  --save_fasta          Include to save a fasta file of the input pdb.
  --conf_filter CONF_FILTER
                        (float, [0-1]) If specified, only domains with a pIoU above this threshold will be saved.
  --plddt_filter PLDDT_FILTER
                        (float, [0-100]) If specified, only domain with a plDDT above this threshold will be saved. Note: if used on a non-AF structure, this will correspond to crystallographic b-factors.
  --iterate             If used, domains under a length threshold (default: 200 residues) will be re-segmented.
  --max_iterations MAX_ITERATIONS
                        (int [1, inf]) Specify the maximum number of re-segmentations that can occur.
  --shuffle_indices     Shuffle domain indices - increases contrast between domain colours in PyMOL.
```

## Outputs

#### String output

By default, Merizo will only output a summary of the parsed PDB file which is printed to the terminal:

```
filename nres nres_dom nres_ndr ndom pIoU runtime result
2xdqA.pdb	425	425	0	3	0.98407	1.89492	6-18_296-459,19-156,157-162_192-295
```
Residue ranges are delimited by `-`, e.g. residues 1 to 200 are `1-200`. Separate domains are delimited by `,`, e.g. `1-200,201-400`, and discontinuous segments are delimited by `_`, e.g. `1-200,201-300_350-400`.

`nres`, `nres_dom` and `nres_ndr` are the total number of residues, number of residues in domains and number of non-domain residues respectively. 

If processing a large number of files, it may be useful to redirect the output into a file using `>` for reference. This can be performed by:
```
python pred.py -d cpu -i examples/*.pdb > examples/results.txt
```

#### PDB outputs

PDB files are generated when using either the `--save_pdb` or `--save_domains` flags.
If the `--save_pdb` flag is used, the original PDB with the domain assignment will be saved with a `.pdb2` file extension. The domain index of each residue is written to the occupancy column. Values in the b-factor column (used for plDDT in AlphaFold2) will be preserved. NDRs are assigned an occupancy of `0.0`.

```
ATOM      1  N   PRO A   6      -8.641  99.395  86.382  1.00     0
ATOM      2  CA  PRO A   6      -9.798 100.225  85.975  1.00     0
ATOM      3  C   PRO A   6      -9.490 101.162  84.771  1.00     0
ATOM      4  O   PRO A   6     -10.415 101.802  84.223  1.00     0
...
ATOM    217  N   GLN A  32      12.896  69.675  61.036  2.00     0
ATOM    218  CA  GLN A  32      13.321  68.978  62.230  2.00     0
ATOM    219  C   GLN A  32      12.070  68.331  62.894  2.00     0
ATOM    220  O   GLN A  32      12.125  67.232  63.353  2.00     0
```

If the `--save_domains` flag is used, a PDB file of each domain structure will be saved separately, as well as one file for the entire structure. The `--conf_filter` and `--plddt_filter` flags can be used to designate pIoU and plDDT thresholds to use for the saved files. Domains with pIoU lower than the value provided by `--conf_filter` or plDDT lower than the value provided by `--plddt_filter` will not be saved.

#### PDF outputs

If the `--save_pdf` option is provided, the domain map of the assignment will be saved in a graphical format. This will generate a single `.pdf` file that contains both the predicted domain map as well as a pIoU map which describes the network confidence for each domain.

### Visualisation

We recommend using PyMOL for visualising the PDB outputs of Merizo. When the `.pdb2` file is opened in PyMOL, the domain assignment can be visualised using the `spectrum q` command in PyMOL applies colouring to each domain to help differentiate them. 

Tips:
- When a larger number of domains are in the structure, the contrast between colours used by `spectrum q` may not be very high, making neighbouring domains hard to differentiate between. Using the `--shuffle_indices` flag in Merizo may help increase the contrast between different domains/colours. 
- NDRs are assigned to domain index 0. After `spectrum q`, their positions will always be in dark blue. To manually apply a colour to NDRs, the `color white, q < 1` command can be used to colour all NDRs white. 
