import numpy as np
import sys

def main():
    # if len(sys.argv) < 10:
    #     print(f"Error, there are missing arguments. The program takes at least {9} arguments, received {len(sys.argv) - 1}...", file=sys.stderr)
    #     return 1
    
    # filepath = sys.argv[1]
    # num_sources = int(sys.argv[2])
    # right_ascension = float(sys.argv[3])
    # declination = float(sys.argv[4])
    # fov_deg = float(sys.argv[5])
    # dynamic_range = float(sys.argv[6])
    # seed = int(sys.argv[7])

    filepath = "/home/joaoc/Documents/data/oskar/skymodels/multiple_gaussian_sources.osm"
    num_sources = 100
    right_ascension = -120
    declination = -60
    fov_deg = 1.0
    dynamic_range = 10000
    majoraxis_max = 600
    minoraxis_max = 500
    seed = 10

    generate_random_gaussian_sources_skymodel(filepath, num_sources, right_ascension, declination, fov_deg, dynamic_range, majoraxis_max, minoraxis_max, seed)

    return 0

    
    

def generate_random_gaussian_sources_skymodel(filepath, num_sources, right_ascension, declination, fov_deg, dynamic_range, majoraxis_max, minoraxis_max, seed):
    rng = np.random.default_rng(seed)

    ra_sources = rng.uniform(right_ascension - fov_deg / 2, right_ascension + fov_deg / 2, num_sources)
    dec_sources = rng.uniform(declination - fov_deg / 2, declination + fov_deg / 2, num_sources)
    intensity_sources = rng.uniform(1, dynamic_range, num_sources)

    majoraxis_sources = rng.uniform(0, majoraxis_max, num_sources)
    minoraxis_sources = rng.uniform(0, minoraxis_max, num_sources)
    rotation_sources = rng.uniform(-180, 180, num_sources)

    # file contents with explanation header
    file_contents = ["#  RA,      Dec,    I,    Q,  U,  V,  freq0,  spix,  RM,   maj,  min,  pa\n",
                     "# (deg),  (deg),  (Jy),                (Hz),   (-),        (\"), (\"), (deg)\n\n"]


    for s in range(num_sources):
        file_contents.append(f"{ra_sources[s]}  {dec_sources[s]}  {intensity_sources[s]}  0  0  0  1.4e9  -0.7  0.0  {majoraxis_sources[s]}  {minoraxis_sources[s]}  {rotation_sources[s]}\n")


    with open(filepath, "wt") as fd:
        fd.writelines(file_contents)


if __name__ == "__main__":
    sys.exit(main())