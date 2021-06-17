bsub -W 12:00 -R "model=XeonGold_5118 rusage[mem=2048]" -N profiling/valgrind.sh
