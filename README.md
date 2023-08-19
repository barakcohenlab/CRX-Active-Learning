# CRX-Active-Learning
Deep learning model for the regulatory grammar of CRX-dependent enhancers and silencers and code to reproduce analyses related to the manuscript.

## Folder organization
- `Data`
    - `activity_summary_stats_and_metadata.txt`: The final, processed activity scores for every sequence measured across all libraries, as well as summary statistics, annotations on how each sequence was created, and assignments to various rounds of learning.
    - `modeling_splits.txt`: A matrix used during classifier training to determine which data batches are "old" (existing training data) and the newly generated data.
    - `processing_config.yml`: Configuration file for how the counts data should be processed into activity scores.
    - `Downloaded`
        - `eLifeMotifs.meme`: The 8 PWMs used in [Friedman *et al.*, 2021](https://elifesciences.org/articles/67403).
        - The processing script also expects data from [Shepherdson *et al.*, 2023](https://www.biorxiv.org/content/10.1101/2023.05.27.542576v1.full) as `retinopathy_data.parquet` and `retinopathy_metadata.txt`. Relevant files can be obtained from the [repo](https://github.com/barakcohenlab/retinopathy-manuscript) for that publication.
    - `Measurements`
        - All raw barcode counts are in subdirectories called `<Library name>/Samples/`. Each sample has a `.counts` file. The directory structure gets filled with intermediates during data processing.
        - `joined_qc_metrics.txt`: Supplementary Table 5.
    - `Sequences`
        - `BarcodeMapping`: files associating barcodes with library members for each assayed library.
        - `Fasta`: FASTA files with the contents of each assayed library and subsets relevant for modeling.
    - `Swaps`: Supplementary Tables 6-8.
- `Figures`: All figures generated by Jupyter notebooks get output here.
- `RegressionModel/best_model.pth.tar`: The parameters for our final regression model.
- `src`
    - Scripts for processing the data, training machine learning models, and performing Global Importance Analysis. Detailed below.
    - `mpra_tools`: Various helper functions for processing, transforming, and visualizing data. With a little bit of work, this could become a standalone package.
    - `selene_files`
        - `enhancer_model.py`: The architecture of our CNN classifier.
        - `enhancer_resnet_regression.py`: The architecture of our final CNN regression model.
        - `metrics.py`: Wrapper functions to evaluation criteria.
- Jupyter notebooks are detailed below and are used to generate figures.

## Reproduciting the results
### 1. Setup the environment
All scripts and notebooks were run on a cluster managed by SLURM.

1. Install miniconda or anaconda, or run `eval $(spack load --sh miniconda3)` on the cluster.
2. Clone this repository.
3. Run `conda env create -f conda-env.yml`. It is recommended to create the environment with a GPU enabled. This environment includes [our fork](https://github.com/rfriedman22/selene) of the Selene package.
4. Activate the environment: `source activate active-learning`
5. Install our extension of the Generic String Kernel:
```sh
git clone https://github.com/barakcohenlab/preimage.git
cd preimage
python setup.py build_ext -i
pip install -e .
```
### 2. Run scripts
Run all Python scripts by submitting the shell scripts to the cluster with `sbatch --mail_user=<user@example.com> script.sh`. You can use the `--dependency=afterok:<job ID>` to setup job dependencies.

1. `process_join_annotate_counts.sh`: Process each MPRA sublibrary from raw barcode counts to activity scores, join all sublibraries together, and annotate for metadata. Estimated runtime: 20-30 minutes.
2. Fit machine learning models (can be done in parallel):
    - `svm_multiclass_cross_validation.sh`: Train SVM classifiers on different rounds of learning. The estimated runtime of this script is ~4 days, but most of that time is spent computing the Gram matrix (AKA the kernel function). The machine learning itself takes 1-2 hours. Once the Gram matrix has been computed, it is saved to file and can be reloaded in subsequent runs with `svm_multiclass_cross_validation.sh path/to/matrix/`.
    - `cnn_multiclass_cross_validation.sh`: Train CNN classifiers on different rounds of learning. Then train a CNN on all of Round 3, instead of with cross-validation. Estimated runtime: 24-36 hours.
    - `fit_summarize_regression_starts.sh`: Train the final CNN regression model on 20 different random initializations, determine which initialization is the best, and then evaluate on the test sets. Do not use `sbatch` to run this script, instead use `sh` since it is a wrapper for two separate `sbatch` calls. Estimated runtime: 30-60 minutes.
3. Once the regression CNN is done, perform Global Importance Analysis. All of these scripts can be run in parallel:
    - `importance_analysis.sh`: Predict the effect of various combinations of motifs at fixed positions. Estimated runtime: 8-12 hours.
    - `importance_analysis_crx_nrl.sh`: Predicts the effect of CRX and NRL motifs at all possible positions. Estimated runtime: anywhere from 2 hours to 24 hours. This variability may depend on how resources are being shared on the GPUs.
    - `importance_analysis_crx_anytf.sh`: Same but with GFI1 instead of NRL. Estimated runtime: anywhere from 2 hours to 24 hours. This variability may depend on how resources are being shared on the GPUs.

### 3. Run all notebooks
These Jupyter notebooks all contain documentation on what they do to transform and visualize data for creating figures.

## Loading the model
If you want to use our regression CNN, use the following Python code:
```python
import src.mpra_tools.loaders
from src.selene_files.enhancer_resnet_regression import EnhancerResnet
model = loaders.load_cnn(
    "RegressionModel/best_model.pth.tar",
    EnhancerResnet(164),
)
```
Our CNN takes a batch of one-hot encoded 164-bp DNA as input and outputs the predicted *cis*-regulatory activity of that sequence in P8 mouse retinas, relative to the basal *Rhodopsin* promoter. You can prepare a list of DNA sequences using the `mpra_tools.modeling.one_hot_encode` function.

## Figure index
Extended Data Figures are prefixed with an S in the below table.
| Notebook name | Figures |
| -- | -- |
| `classifier_performance.ipynb` | 1c, 2a, 2b, 2c, S2, S3 |
| `confident_predictions.ipynb` | 1f |
| `data_coverage.ipynb` | 2d |
| `enhancer_mutagenesis.ipynb` | All of Figure 4, S6, S7 |
| `filter_svm_performance.ipynb` | S1 |
| `motif_importance_analysis.ipynb` | 3a, 3b, 3c, 3d, 3e, S4, S5 |
| `spacing_analysis.ipynb` | 3f, 3g |
| `src/eval_regression_on_test_sets.py` | 1d, 1e |