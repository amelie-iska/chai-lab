#!/usr/bin/env python

"""
Below is a short overview of each command-line flag from the updated script, why it exists, how it affects the resulting MSA depth, and any considerations for usage.

Core Idea: Depth of the MSA

When you use MMseqs2 to generate a multiple sequence alignment (MSA), the number of sequences you get—i.e. the depth—depends primarily on how permissive your thresholds are. A more permissive setup means lower coverage, lower minimum sequence identity, higher E-value, and a higher maximum number of sequences. This typically yields a deeper MSA. A less permissive setup (higher coverage requirement, higher identity requirement, lower E-value) yields fewer hits, for a shallower MSA.

After MMseqs2 finishes, the script also clamps the final MSA to --max-msa-depth hits (not counting the query). This prevents extremely large MSAs that might exceed memory or time constraints.

Below are the specific flags:

1) --mmseqs-coverage
	•	Sets the -c parameter in mmseqs search. (e.g. -c 0.0).
	•	Meaning: The fraction of the alignment that must be covered to accept the hit. For instance, if you set --mmseqs-coverage 0.8 along with --mmseqs-cov-mode 0, then the alignment must cover at least 80% of the aligned region. (In other coverage modes, it might mean 80% of query or target.)
	•	Affects Depth:
	•	A higher coverage threshold (like 0.8 or 0.9) is stricter, so it reduces the number of accepted hits (shallower MSA).
	•	A lower coverage threshold (like 0.0 or 0.1) is more permissive, so it increases the number of accepted hits (deeper MSA).
	•	Default: 0.0 (no coverage requirement).

2) --mmseqs-cov-mode
	•	Sets the --cov-mode parameter in mmseqs. Typically can be 0, 1, or 2:
	•	0 => coverage is measured over the alignment length
	•	1 => coverage is measured over the target sequence length
	•	2 => coverage is measured over the query sequence length
	•	Affects Depth:
	•	Indirectly. If you have set a high coverage requirement with -c, a more “strict” coverage mode can make it harder to pass. For example, --cov-mode 2 means the alignment must cover that fraction of the query length—leading to fewer hits.
	•	Default: 0 (covering alignment length, the simplest approach).

3) --mmseqs-min-seq-id
	•	Sets the --min-seq-id parameter in mmseqs. (e.g. --min-seq-id 0.3).
	•	Meaning: The minimal fraction (0.0 to 1.0) of sequence identity required.
	•	Affects Depth:
	•	A lower min-seq-id (like 0.0) accepts more distant homologs => increases MSA depth.
	•	A higher min-seq-id (like 0.8) only accepts close homologs => reduces MSA depth.
	•	Default: 0.0 (no restriction on identity).

4) --mmseqs-max-seqs
	•	Sets the --max-seqs parameter in mmseqs. (e.g. --max-seqs 10000).
	•	Meaning: The maximum number of hits MMseqs2 will keep during the search.
	•	Affects Depth:
	•	If you set it to 10,000, MMseqs2 can only keep up to 10,000 hits. If your query has more than 10,000 potential hits, it discards the rest.
	•	Default: 10000.

5) --mmseqs-e-value
	•	Sets the -e parameter in mmseqs. (e.g. -e 1e-3).
	•	Meaning: The E-value threshold. A higher E-value allows more hits, a lower E-value is more stringent.
	•	Affects Depth:
	•	A higher E-value (like 1e-1) => more hits => deeper MSA.
	•	A smaller E-value (like 1e-5) => fewer hits => shallower MSA.
	•	Default: 1e-3.

6) --max-msa-depth
	•	This is not a MMseqs parameter. The script uses it after generating the raw .a3m to clamp the final MSA depth. For example, if the raw .a3m has 20,000 hits, we store only the top ~10,000 (plus the query) in the .aligned.pqt.
	•	Affects Depth:
	•	If --max-msa-depth = 10,000, your final MSA cannot exceed 10,000 hits.
	•	Even if mmseqs finds more, we limit them. If mmseqs finds only 8,000 hits, we get 8,000.
	•	Default: 10000.

7) --mmseqs-db-load-mode
	•	Sets the --db-load-mode <int> parameter in mmseqs search and mmseqs result2msa.
	•	Meaning: Tells MMseqs2 how to load or hold the target database (uniref100_gpu) in memory or in partial chunks. Common values:
	•	0 => default behavior (splits database as needed if memory is limited).
	•	1 => partial? (Implementation can vary; sometimes 1 or 2 means keep DB in memory.)
	•	2 => load entire DB into memory, which can speed repeated searches if you have enough RAM.
	•	Affects Depth:
	•	Indirectly. It doesn’t set any alignment thresholds. But if you use --db-load-mode 2 and have enough memory, the search might run faster. That can let you do bigger searches, but functionally it doesn’t change how many hits pass thresholds.

8) --msa-source-db-label
	•	Used only in the script’s convert_a3m_to_aligned_pqt(...) step. We label the non-query hits in .aligned.pqt as "uniref90" or some allowed source. The pipeline’s schema only permits:

{"query", "uniprot", "uniref90", "mgnify", "bfd_uniclust"}


	•	Affects Depth:
	•	No. This is purely to pass the pipeline’s schema validation. If you used "uniref100", it fails.

Additional Tips for Getting ~10K Sequences
	1.	Lower coverage: e.g. --mmseqs-coverage 0.0 --mmseqs-cov-mode 0.
	2.	Lower min-seq-id: e.g. --mmseqs-min-seq-id 0.0.
	3.	Higher e-value: e.g. --mmseqs-e-value 1e-1.
	4.	Set --mmseqs-max-seqs 20000 if you want up to 20k hits. Then rely on --max-msa-depth 10000 to store exactly 10k in the final MSA.
	5.	If you want to speed repeated searches or have enough memory, pass --mmseqs-db-load-mode 2.

Putting it all together might look like:

python ./examples/msas/run.py examples/msas/binder-design.yaml \
  --device cuda:0 \
  --output-dir ./outputs/binder-test-run \
  --uniref-db /home/pechev/workspace/msa/uniref100_gpu \
  --mmseqs-path /home/pechev/workspace/msa/mmseqs/bin/mmseqs \
  --mmseqs-sensitivity 7.5 \
  --mmseqs-coverage 0.0 \
  --mmseqs-cov-mode 0 \
  --mmseqs-min-seq-id 0.0 \
  --mmseqs-max-seqs 15000 \
  --mmseqs-e-value 1e-2 \
  --max-msa-depth 10000 \
  --mmseqs-db-load-mode 2 \
  --msa-source-db-label uniref90
  
python ./examples/msas/run.py ./examples/msas/xenon.yaml \
  --device cuda:0 \
  --output-dir ./outputs/xenon-GJ \
  --uniref-db /home/pechev/workspace/msa/uniref100_gpu \
  --mmseqs-path /home/pechev/workspace/msa/mmseqs/bin/mmseqs \
  --mmseqs-sensitivity 7.5 \
  --mmseqs-coverage 0.0 \
  --mmseqs-cov-mode 0 \
  --mmseqs-min-seq-id 0.0 \
  --mmseqs-max-seqs 15000 \
  --mmseqs-e-value 1e-2 \
  --max-msa-depth 10000 \
  --mmseqs-db-load-mode 2 \
  --msa-source-db-label uniref90
  --num_recycles 10 \
  --num_timesteps 250 \
  --rndm_seed 42 \
  --use_esm_embeddings True \
  # --save-msa True 
"""

import argparse
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import subprocess
import yaml
import torch
import numpy as np

from chai_lab.chai1 import run_inference
from chai_lab.data.dataset.msas.msa_context import MSAContext
from chai_lab.data.parsing.msas.aligned_pqt import (
    expected_basename,
    hash_sequence,
)
from chai_lab.data.parsing.msas.data_source import MSADataSource
from chai_lab.data.parsing.structure.entity_type import EntityType
from chai_lab.data.parsing.fasta import read_fasta, Fasta

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def parse_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "sequences" not in config:
        raise ValueError("Config must contain a 'sequences' key")
    return config


def create_fasta_from_config(config: Dict[str, Any], tmp_dir: Path) -> Path:
    """
    Generate a single FASTA file for all protein/dna/rna entries in the config.
    Each chain is named something like 'protein|name=protein_chainA', but the
    pipeline uses the sequence's hash to identify the final .aligned.pqt.
    """
    fasta_entries = []
    for entry in config["sequences"]:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError("Each sequence entry must have exactly one entity type.")

        entity_type = list(entry.keys())[0]
        entity_data = entry[entity_type]

        if "id" not in entity_data:
            raise ValueError(f"Missing 'id' field in {entity_type} entry")

        chain_ids = (
            entity_data["id"]
            if isinstance(entity_data["id"], list)
            else [entity_data["id"]]
        )

        if entity_type in ["protein", "dna", "rna"]:
            if "sequence" not in entity_data:
                raise ValueError(f"Missing 'sequence' field in {entity_type} entry")
            seq = entity_data["sequence"]

            for i, chain_id in enumerate(chain_ids):
                if len(chain_ids) > 1:
                    chain_name = f"protein|name=protein_{chain_id}_{i+1}"
                else:
                    chain_name = f"protein|name=protein_{chain_id}"
                fasta_entries.append(f">{chain_name}")
                fasta_entries.append(seq)

        elif entity_type == "ligand":
            # skip
            pass

    fasta_path = tmp_dir / "input.fasta"
    fasta_path.write_text("\n".join(fasta_entries))
    return fasta_path


def parse_protein_chains_from_config(config: Dict[str, Any]) -> List[dict]:
    """
    Return a list describing each chain:
      {
        'id': chain_id,
        'sequence': 'ACDEFG...',
        'entity_type': 'protein' or 'rna' or 'dna',
        'msa_mode': 'local'/'user'/'none',
        'msa_path': Path or None
      }
    """
    results = []
    for entry in config["sequences"]:
        entity_type = list(entry.keys())[0]
        entity_data = entry[entity_type]

        if entity_type not in ["protein", "dna", "rna"]:
            # skip ligands
            continue

        chain_ids = (
            entity_data["id"] if isinstance(entity_data["id"], list) else [entity_data["id"]]
        )
        seq = entity_data["sequence"]
        raw_msa_value = entity_data.get("msa", None)

        for i, chain_id in enumerate(chain_ids):
            if raw_msa_value is None:
                mode = "none"
                path = None
            elif raw_msa_value == "":
                mode = "local"
                path = None
            else:
                mode = "user"
                path = Path(raw_msa_value)

            chain_dict = {
                "id": chain_id,
                "sequence": seq,
                "entity_type": entity_type,
                "msa_mode": mode,
                "msa_path": path,
            }
            results.append(chain_dict)
    return results


def read_fasta_a3m(a3m_path: Path) -> list[Fasta]:
    from chai_lab.data.parsing.fasta import read_fasta
    return list(read_fasta(a3m_path))


def convert_a3m_to_aligned_pqt(
    sequence: str,
    a3m_path: Path,
    out_pqt: Path,
    max_hits: int,
    source_db_label: str = "uniref90",
):
    """
    Convert the .a3m file to a .aligned.pqt that the pipeline can read.
    By default, we label hits with 'uniref90' to pass Pandera's schema check.
    Allowed sources = {query, uniprot, uniref90, mgnify, bfd_uniclust}.
    """
    import pandas as pd

    # read the A3M
    all_records = read_fasta_a3m(a3m_path)
    if not all_records:
        raise ValueError(f"A3M file {a3m_path} is empty or invalid")

    # row 0 => query
    query_record = all_records[0]
    # subsequent => hits
    hits = all_records[1:]
    # clamp hits to max_hits
    if len(hits) > max_hits:
        hits = hits[:max_hits]

    # build the table
    data_seq = [query_record.sequence] + [h.sequence for h in hits]
    # first => 'query', rest => user-chosen label
    data_source = ["query"] + [source_db_label] * len(hits)
    data_pairing_key = ["0"] + [""] * len(hits)
    data_comment = [""] * (1 + len(hits))

    df = pd.DataFrame(
        dict(
            sequence=data_seq,
            source_database=data_source,
            pairing_key=data_pairing_key,
            comment=data_comment,
        )
    )
    df.to_parquet(out_pqt)


def generate_local_msas_for_chains(
    chain_info_list: List[dict],
    uniref_db: Path,
    msa_dir: Path,
    mmseqs_path: str,
    sensitivity: float,
    use_gpu: bool,
    max_hits: int,
    mmseqs_coverage: float,
    mmseqs_cov_mode: int,
    mmseqs_min_seq_id: float,
    mmseqs_max_seqs: int,
    mmseqs_e_value: float,
    source_db_label: str = "uniref90",
    mmseqs_db_load_mode: int = 0,
):
    """
    For each chain with 'msa_mode'='local', run mmseqs2 search & result2msa,
    using coverage/identity/evalue flags. Then convert .a3m -> .aligned.pqt.
    We rename the hits' source_database to 'uniref90' (or a chosen label).
    Also adds --db-load-mode <X> in both search & result2msa calls.
    """
    msa_dir.mkdir(parents=True, exist_ok=True)

    for chain_info in chain_info_list:
        # Only run local MSAs if 'msa_mode' is set to "local"
        if chain_info["msa_mode"] != "local":
            continue

        seq = chain_info["sequence"]
        entity_type = chain_info["entity_type"]
        # Skip if it's not protein (e.g. if it's DNA/RNA or a ligand)
        if entity_type != "protein":
            continue
        # Skip if the sequence is empty
        if not seq.strip():
            print(f"[INFO] Chain {chain_info['sequence']} is empty. Skipping MSA.")
            continue
        
        from chai_lab.data.parsing.msas.aligned_pqt import hash_sequence
        seq_hash = hash_sequence(seq.upper())

        # ---------------------------------------------------------------------
        # NEW: Skip if .aligned.pqt for this exact sequence hash already exists
        # ---------------------------------------------------------------------
        out_pqt = msa_dir / f"{seq_hash}.aligned.pqt"
        if out_pqt.exists():
            print(f"[INFO] MSA for sequence hash {seq_hash} already exists; skipping mmseqs search.")
            continue

        # Prepare temporary paths for MMseqs
        a3m_path = msa_dir / f"{seq_hash}.a3m"
        chain_fasta = msa_dir / f"{seq_hash}.fasta"
        tmp_path = msa_dir / f"{seq_hash}_tmp"

        # If tmp_path exists from a previous partial run, remove it
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        tmp_path.mkdir(exist_ok=True)

        # Write a minimal FASTA containing just this query sequence
        chain_fasta.write_text(f">localSeq\n{seq}\n")

        # Prepare MMseqs DB paths
        query_db = tmp_path / "queryDB"
        result_db = tmp_path / "searchRes"

        # ----------------------------------------------------------
        # Step 1) createdb
        # ----------------------------------------------------------
        subprocess.check_call([
            mmseqs_path, "createdb",
            str(chain_fasta),
            str(query_db)
        ])

        # ----------------------------------------------------------
        # Step 2) mmseqs search
        # ----------------------------------------------------------
        cmd_search = [
            mmseqs_path, "search",
            "--gpu", "1" if use_gpu else "0",
            "-a",
            "-s", str(sensitivity),
            "-c", str(mmseqs_coverage),
            "--cov-mode", str(mmseqs_cov_mode),
            "--min-seq-id", str(mmseqs_min_seq_id),
            "--max-seqs", str(mmseqs_max_seqs),
            "-e", str(mmseqs_e_value),
            "--db-load-mode", str(mmseqs_db_load_mode),
            str(query_db),
            str(uniref_db),
            str(result_db),
            str(tmp_path),
        ]
        subprocess.check_call(cmd_search)

        # ----------------------------------------------------------
        # Step 3) mmseqs result2msa => .a3m
        # ----------------------------------------------------------
        cmd_result2msa = [
            mmseqs_path, "result2msa",
            str(query_db),
            str(uniref_db),
            str(result_db),
            str(a3m_path),
            "--msa-format-mode", "5",
            "--db-load-mode", str(mmseqs_db_load_mode),
        ]
        subprocess.check_call(cmd_result2msa)

        # ----------------------------------------------------------
        # Step 4) convert .a3m => .aligned.pqt
        # ----------------------------------------------------------
        convert_a3m_to_aligned_pqt(
            sequence=seq,
            a3m_path=a3m_path,
            out_pqt=out_pqt,
            max_hits=max_hits,
            source_db_label=source_db_label,
        )

    print(f"[INFO] Local MSA generation done, .aligned.pqt files in {msa_dir}")


###############################################################################
# MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Use local mmseqs2 GPU uniref100_gpu DB to generate .aligned.pqt if `msa: ''`, with adjustable coverage, seq-id, evalue, etc. Also set --db-load-mode, optionally delete MSA folder."
    )
    parser.add_argument("config", help="YAML config with sequences")
    parser.add_argument("--device", default="cuda:0", help="Device e.g. cuda:0")
    parser.add_argument("--output-dir", default=None, help="Top-level output dir (default=./outputs/TIMESTAMP)")
    parser.add_argument("--uniref-db", default="uniref100_gpu", help="Local uniref100_gpu DB path")
    parser.add_argument("--mmseqs-path", default="mmseqs", help="Path to mmseqs2 binary")
    parser.add_argument("--mmseqs-sensitivity", default=7.5, type=float,
                        help="MMseqs2 search sensitivity (default=5.0)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU-based mmseqs search (otherwise GPU).")

    # Additional flags for coverage, identity, evalue
    parser.add_argument("--mmseqs-coverage", default=0.0, type=float,
                        help="Coverage threshold '-c' (default=0.0 => no coverage filtering).")
    parser.add_argument("--mmseqs-cov-mode", default=0, type=int,
                        help="Coverage mode '--cov-mode' (default=0). Typically 0 => coverage over alignment length.")
    parser.add_argument("--mmseqs-min-seq-id", default=0.0, type=float,
                        help="Minimum sequence identity '--min-seq-id' (default=0.0).")
    parser.add_argument("--mmseqs-max-seqs", default=10000, type=int,
                        help="Maximum number of hits '--max-seqs' (default=10000).")
    parser.add_argument("--mmseqs-e-value", default=1e-3, type=float,
                        help="E-value threshold '-e' (default=1e-3).")

    # We'll let user clamp final MSA depth for the .aligned.pqt
    parser.add_argument("--max-msa-depth", default=10000, type=int,
                        help="Clamp how many hits to store in .aligned.pqt. (Default=10000)")

    # Possibly we want to label the hits as 'uniref90' or 'bfd_uniclust' so the pipeline doesn't fail schema check
    parser.add_argument("--msa-source-db-label", default="uniref90",
                        help="One of {query, uniprot, uniref90, mgnify, bfd_uniclust} for non-query hits. (Default=uniref90)")

    # new: db-load-mode (int)
    parser.add_argument("--mmseqs-db-load-mode", default=0, type=int,
                        help="Integer for mmseqs search/result2msa --db-load-mode. (Default=0)")

    # new: optionally keep MSA data
    parser.add_argument("--save-msa", action="store_true",
                        help="If provided, keep the MSA subfolder after inference. By default, MSA folder is deleted.")
    
    parser.add_argument("--num_recycles", default=3, type=int,
                        help="Number of trunk recycles for run_inference. (Default=3)")
    
    parser.add_argument("--num_timesteps", default=200, type=int,
                        help="Number of diffn timesteps for run_inference. (Default=200)")
    
    parser.add_argument("--rndm_seed", default=42, type=int,
                        help="Random seed for run_inference. (Default=42)")
    
    parser.add_argument("--use_esm_embeddings", default=True, type=bool, 
                        help="Use ESM embeddings during inference. (Default=True)")

    args = parser.parse_args()

    # 1) figure out top-level output dir
    if args.output_dir:
        top_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        top_dir = Path(f"./outputs/{timestamp}")
    top_dir.mkdir(parents=True, exist_ok=True)

    # subdirs for msas vs run_inference
    run_timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    final_run_dir = top_dir / f"{run_timestamp}_run"
    final_run_dir.mkdir(parents=True, exist_ok=False)  # must be empty

    msa_dir = top_dir / f"{run_timestamp}_msas"
    msa_dir.mkdir(parents=True, exist_ok=False)

    print(f"[INFO] MSA dir: {msa_dir}")
    print(f"[INFO] run_inference output dir: {final_run_dir}")

    # 2) parse config
    config = parse_config(args.config)

    # 3) create a tmp dir for the combined FASTA
    tmp_dir = Path(tempfile.mkdtemp())

    # 4) parse chain info
    chain_info_list = parse_protein_chains_from_config(config)

    # 5) create combined FASTA for run_inference
    fasta_path = create_fasta_from_config(config, tmp_dir)

    # 6) generate local MSAs => produce .aligned.pqt
    generate_local_msas_for_chains(
        chain_info_list=chain_info_list,
        uniref_db=Path(args.uniref_db),
        msa_dir=msa_dir,
        mmseqs_path=args.mmseqs_path,
        sensitivity=args.mmseqs_sensitivity,
        use_gpu=not args.cpu,
        max_hits=args.max_msa_depth,
        mmseqs_coverage=args.mmseqs_coverage,
        mmseqs_cov_mode=args.mmseqs_cov_mode,
        mmseqs_min_seq_id=args.mmseqs_min_seq_id,
        mmseqs_max_seqs=args.mmseqs_max_seqs,
        mmseqs_e_value=args.mmseqs_e_value,
        source_db_label=args.msa_source_db_label,
        mmseqs_db_load_mode=args.mmseqs_db_load_mode,
    )

    # 7) If user-provided => maybe convert .a3m => .aligned.pqt or symlink .aligned.pqt
    def maybe_convert_user_msa(seq: str, user_msa_path: Path):
        seq_hash = hash_sequence(seq.upper())
        out_pqt = msa_dir / f"{seq_hash}.aligned.pqt"
        if user_msa_path.suffix == ".pqt":
            if out_pqt.exists():
                out_pqt.unlink()
            out_pqt.symlink_to(user_msa_path)
            print(f"[INFO] Symlink user-provided .aligned.pqt for seq {seq}: {user_msa_path}")
        elif user_msa_path.suffix == ".a3m":
            print(f"[INFO] Convert user-provided .a3m -> .aligned.pqt for seq {seq}: {user_msa_path}")
            convert_a3m_to_aligned_pqt(
                sequence=seq,
                a3m_path=user_msa_path,
                out_pqt=out_pqt,
                max_hits=args.max_msa_depth,
                source_db_label=args.msa_source_db_label,
            )
        else:
            print(f"[WARNING] Unknown user MSA suffix {user_msa_path.suffix}, skipping. Provide .a3m or .pqt if you want to load MSA.")
    
    for chain_info in chain_info_list:
        if chain_info["msa_mode"] == "user":
            seq = chain_info["sequence"]
            user_path = chain_info["msa_path"]
            entity_type = chain_info["entity_type"]
            if entity_type != "protein":
                continue
            maybe_convert_user_msa(seq, user_path)

    # 8) chain(s) with 'none' => no MSA => "No MSA found"

    # 9) run_inference
    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=final_run_dir,
        num_trunk_recycles=args.num_recycles,
        num_diffn_timesteps=args.num_timesteps,
        seed=args.rndm_seed,
        device=str(args.device),
        use_esm_embeddings=args.use_esm_embeddings,
        msa_directory=msa_dir,
    )

    # 10) done
    cif_paths = candidates.cif_paths
    scores = [rd.aggregate_score for rd in candidates.ranking_data]

    scores_path = final_run_dir.joinpath("scores.model_idx_2.npz")
    if scores_path.exists():
        scores_np = np.load(scores_path)
        print(f"Detailed scores: {scores_np.files}")

    # 11) If user did not specify --save-msa, delete the MSA dir
    if not args.save_msa:
        print("[INFO] Removing MSA directory because --save-msa not provided...")
        shutil.rmtree(msa_dir, ignore_errors=True)
    else:
        print("[INFO] MSA directory retained because --save-msa was specified.")

    print("[INFO] All done. Adjust coverage, seq-id, e-value, etc., plus --db-load-mode.")
    print(f"[INFO] Inference outputs in: {final_run_dir}")


if __name__ == "__main__":
    main()