
## Run
Python 3.10+; install `pip install -r requirements.txt`.

```
python run.py --records data/records.csv --embeddings data/embeddings.npz --train-end YYYY-MM-DD --val-end YYYY-MM-DD --out results
```
Replace dates with documented chronological cutoffs. Training and validation records with outcomes crossing their respective cutoff are excluded. Test outcomes must be observed. Embedding IDs must match CSV order exactly; store string arrays, not pickled objects. See schema.md.

