# CA-MTF reproduction implementation

Implements the evaluated architecture described in the supplied manuscript, not the proposed-only BERT/SigLIP or observed temporal graph. No generated benchmark data or fabricated metrics are included.

## Run
Python 3.10+; install `pip install -r requirements.txt`.

```
python run.py --records data/records.csv --embeddings data/embeddings.npz --train-end YYYY-MM-DD --val-end YYYY-MM-DD --out results
```
Replace dates with documented chronological cutoffs. Training and validation records with outcomes crossing their respective cutoff are excluded. Test outcomes must be observed. Embedding IDs must match CSV order exactly; store string arrays, not pickled objects. See schema.md.

## Decisions not specified by paper
Default 5 seeds 42–46, maximum 100 epochs, threshold 0.5, Ridge and balanced logistic regression, sklearn default histogram boosting, bootstrap by brand (1000 replicates per seed). These are configurable or replaceable reconstruction choices, not claimed original settings. Scalar text/image inner product is an additional gate input. Structured missing indicators augment the 11 features; unknown categories and years use zero embedding. Unknown legal inputs are median-imputed with missing indicators on the structured branch; rule ranking excludes pairs with unknown rule inputs. No post-hoc calibration. No economics ablation retains the separately defined rule branch and latent effects, and excludes records without any semantic modality (compare sample counts). Value head follows the equation using fused representation only. PR-AUC is average precision; recall at FPR 0.20 is the maximum empirical ROC recall under that bound, a descriptive metric not an operational threshold. Missingness independently removes text/image at inference; existing missingness remains. All preprocessing fitting uses training only. Validation checkpoint selection uses MAE within 1% of best, then AUROC and recall. Blends fixed to manuscript coefficients; their original validation selection cannot be verified. Training saves weights and predictions, but preprocessing is rebuilt from original inputs when rerunning; checkpoints alone are not deployment bundles.

## Data limitation
The manuscript leaves outcome definitions, linkage, feature provenance, sample counts and split dates unspecified. Consequently this package implements the model and experiment runner from a verified joined table, not an exact reconstruction of the original dataset. No rule-derived labels are generated. Do not interpret screening predictions as legal violations. Supply independently documented labels and point-in-time embeddings/features, and audit linkage and leakage before reporting metrics. Source URLs in schema.md are those listed in the manuscript, not verified downloads.
