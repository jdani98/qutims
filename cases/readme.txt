This folder contains the scripts and results for the article. Datasets are described in arXiv:2310.20671.
Results are still to be published in a new paper version.
Some of the scripts are prepared to be run in slurm systems and using Dask. The user may need to make some modifications to make the code runnable.


FOLDER NAMES
Each case contains 4 folders:
    - bash0 for 10 optimizations with different initial parameters and numerical gradients without sampling noise (exact).
    - bash0k for an optimization with exact analytical gradients, with the same parameter initialization as best optimisation in bash0 (*).
    - bash1k and bash10k contain 8 different optimisations that use analytical gradients with 1000 and 10000 shots, respectively, with 
    the same parameter initialization as best in bash0 (*).

FILE NAMES
    - PY scripts.
        -> opt_*joba are the scripts for training with qurecnets and using a predefined optimiser. Made to be run in a job array.
        -> visfit_* read a file param_best*.dat and plot the dataset and the prediction of the QRNN with these parameters.
        -> *_loss_curves read a log*.dat file and plot the curves of training and validation RMSE through the optimization.

    - DAT files
        -> param0_* contain the array of initial parameters.
        -> param_best_* contain the array of the best set of parameters after an optimization.
        -> .grad_best_* contain the array of gradients in the last data sample (instance) when the best set of parameters was found during an optimization.
        -> log_* cointain information and the prints of the training and validation RMSEs during an optimization.
    
    - CSV files
        -> .shuffle_indices_* contains the arrays order in which the samples were run every epoch in an optimisation in bash0. This file is then read for the emulations with shots to 
        ensure the same order and make an adequate comparison noiseless vs noisy.
        ->> optimisations_info_* collects the configuration and the results from the optimizations. Columns meaning:
            -- originID: identifier for optimizations initialized and shuffle from a previous seed. bash0k, bash1k and bash10k take the ID to read the param0_* and shuffle_indices_* of
            the best optimization in bash0.
            -- jobID: identifier of the job for these optimisations.
            -- cores, memory, CPU time, time are stats from slurm command seff: "Cores", "Memory utilized", "CPU utilized", "Job Wall-clock time".
            -- Npoints: total number of points of a dataset. Then, 80% are used for training+validation, and the 80% of this set, for training only.
            -- nT, nE, nM, nL, nx are hyperparameters of the QRNN: number of time steps (circuit blocks), exchange qubits, memory qubits, layers and data re-uploadings.
            -- Nwout: number of points to predict each sample.
            -- Nparam: number of trainable parameters.
            -- optimizer: optimization algorithm.
            -- Nepochs: number of epochs until lowest validation RMSE is reached.
            -- Ncev/w: number of circuit evaluations per window (sample) (**).
            -- Ncev: number of circuit evaluations per epoch (**).
            -- grad: whether a gradient is computed through analytical (True) or numerical methods (False).
            -- shuffle: whether or not the samples are run in random order in a single epoch.
            -- stepsize: parameter for the learning rate of Adam.
            -- Nshots: number of shots for optimizations with simulated sampling noise.
            -- RMSE tra, RMSE val, RMSE tes, RMSE ftes: root mean squared-error for training, validation, test and fill-test series (***).
    
    - PNG files: plots

All the output files are identified with a job ID.


(*) Best optimisation is the one that reaches a lower validation Root Mean Squared Error (RMSE) during the training.
(**) A circuit evaluation is defined as a single evaluation of the circuit outputs (expectation values) after many repetitions (shots), for a given set of parameters.
A gradient or Hessian computation often requires many circuit evaluations, as shifts are added to every parameter independently.
(***) Fill-test RMSE considers the predictions to all the points in the test series, by shifting the sample windows Nwout byNwout points.