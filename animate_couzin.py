"""Run the Couzin et al. (2005) baseline on the CB-ELE world; write metrics CSVs and a GIF.

    python animate_couzin.py --output-dir results [--informed 0.3] [--seed 42]
"""

from baseline_animation import main

if __name__ == "__main__":
    main("couzin")
