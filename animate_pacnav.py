"""Run the PACNav baseline (Ahmad et al. 2022) on the CB-ELE world; write metrics CSVs and a GIF.

    python animate_pacnav.py --output-dir results [--informed 0.3] [--seed 42]
"""

from baseline_animation import main

if __name__ == "__main__":
    main("pacnav")
