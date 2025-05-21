Analysis Tools for DNA

To run kinetics data:
1) Format your dataset in the following format:
Time | Dimer #1 | Tile #1 | Dimer #2 | Tile #2 | ...
2) cd into the "kinetics/" folder (cd kinetics)
3) run "python3 run_kinetics_analysis.py {path_to_dataset} {path_to_output}"
For instance, "python3 run_kinetics_analysis.py {./data/exp1} {./results052026}"
