# bo-catalyst
Bayesian Optimization for Catalyst Discovery

<!-- TODO: crisp story + results gif  -->

#### How to use
```python
python -m src.data.fetch_mp -e Ni,Cu --outfile data/raw_mp_thermo.parquet

python -m src.data.fetch_ch \
        --facet "" -e Ni,Cu \
        --no-strict-ads \
        --require-both \
        --element-source either \
        --keep-unknown-comp \
        --outfile data/ch_all_facets_either_keepUnknown_v2.parquet \
        --verbose 
# optional: --reactants "~H" (contains) or "H*" (exact-ish), --chem "NiCu"

python -m src.data.join_clean --ch data/raw_catalysis_hub.parquet --mp data/raw_mp_thermo.parquet --outfile data/processed.parquet

# 1) (optional) regenerate BO dataset if needed
python -m src.features.make_bo_dataset --infile data/processed.parquet --outfile data/bo_dataset.parquet --target-ads 0.0

# 2) Optimize & write suggestions under results/
python -m src.bo.optimize --infile data/bo_dataset.parquet --suggestions 5 --outfile results/bo/suggestions.parquet

# 3) Visualize (figures go to results/figures/)
python -m src.bo.visualize --data data/bo_dataset.parquet --suggestions results/bo/suggestions.parquet --figdir results/figures --target-ads 0.0
```
