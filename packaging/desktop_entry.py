import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from vasp_slurm_agent.desktop import main
    raise SystemExit(main())
