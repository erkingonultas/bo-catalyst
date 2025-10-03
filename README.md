# bo-catalyst
Bayesian Optimization for Catalyst Discovery

<!-- TODO: crisp story + results gif  -->

#### How to use
```python
# Fetch data

python -m src.data.fetch_mp_improved -e Ni,Cu -o data/test_mp.parquet \
    --eah-max 1.0 --include-unstable

python -m src.data.fetch_ch \
        --facet "" -e Ni,Cu \
        --no-strict-ads \
        --require-both \
        --element-source either \
        --keep-unknown-comp \
        --outfile data/raw_catalysis_hub.parquet \
        --verbose 
# optional: --reactants "~H" (contains) or "H*" (exact-ish), --chem "NiCu"

# Combine both to create a dataset
python -m src.data.join_clean --ch data/raw_catalysis_hub.parquet --mp data/raw_mp_thermo.parquet --outfile data/processed.parquet

# 1) (optional) regenerate BO dataset if needed
python -m src.features.make_bo_dataset --infile data/processed.parquet --outfile data/bo_dataset.parquet --target-ads 0.0

# 2) Optimize & write suggestions under results/
python -m src.bo.optimize --infile data/bo_dataset.parquet --suggestions 5 --outfile results/bo/suggestions.parquet

# 3) Visualize (figures go to results/figures/)
python -m src.bo.visualize --data data/bo_dataset.parquet --suggestions results/bo/suggestions.parquet --figdir results/figures --target-ads 0.0
```
