


for dr in 100 300 1000 2000 5000 10000; do
  for seed in $(seq 0 4); do
    for kind in point gaussian; do
      python src/simulation/generate_osm_skymodel.py $kind --num-sources 50 --dr $dr --seed $seed \
        --peak-flux 10.0 --sigma 1e-4 --ra0 -120.0 --dec0 -60.0 --out data/oskar/skymodels/hogbom_experiments/${kind}_dr${dr}_s${seed}.osm
    done
  done
done