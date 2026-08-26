"""
No-Ray replacement for pfb_init.

Calls stokes_vis() directly in a serial loop instead of via Ray remote tasks.
Required because Ray worker IPC fails under Docker + Rosetta 2 (ARM64 host).

Usage (run inside Docker)
-------------------------
docker run --rm --platform linux/amd64 \\
  -v /path/to/repo:/workspace \\
  ghcr.io/ratt-ru/pfb-imaging:latest \\
  python /workspace/scripts/pfb_init_noray.py \\
    --ms /workspace/data/simulated/SCENARIO/obs.ms \\
    --output-prefix /workspace/data/simulated/SCENARIO/obs \\
    --nthreads 4
"""

import argparse
import sys
import time
from pathlib import Path

import fsspec
import numpy as np
from daskms import xds_from_storage_ms as xds_from_ms
from daskms.fsspec_store import DaskMSStore
from ducc0.misc import resize_thread_pool


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ms", required=True, help="Path to the Measurement Set")
    p.add_argument("--output-prefix", required=True,
                   help="Output prefix; obs_I.xds is written at {prefix}.xds")
    p.add_argument("--nthreads", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--precision", default="double", choices=["single", "double"])
    p.add_argument("--product", default="I")
    return p


def main():
    args = get_parser().parse_args()

    # Add pfb_imaging to the path (same dir as our experiment scripts)
    for extra_path in [str(Path(__file__).parent.parent / "src"),
                       str(Path(__file__).parent.parent)]:
        if extra_path not in sys.path:
            sys.path.insert(0, extra_path)

    from pfb_imaging.utils.misc import construct_mappings
    from pfb_imaging.utils.naming import set_output_names
    from pfb_imaging.utils.stokes2vis import stokes_vis  # non-Ray version

    # Note: resize_thread_pool from ducc0 can conflict with Numba's threading init
    # under Rosetta 2 emulation. Skip it here; nthreads only affects Numba parallelism.
    # resize_thread_pool(args.nthreads)

    output_filename, _, log_directory, oname = set_output_names(
        args.output_prefix, args.product, None, None,
    )

    xds_store = DaskMSStore(f"{output_filename}.xds")
    if xds_store.exists():
        if args.overwrite:
            print(f"[pfb_init_noray] Overwriting {output_filename}.xds")
            xds_store.rm(recursive=True)
        else:
            print(f"[pfb_init_noray] {output_filename}.xds exists. Use --overwrite to replace.")
            sys.exit(1)

    fs = fsspec.filesystem(xds_store.protocol)
    fs.makedirs(xds_store.url, exist_ok=True)
    print(f"[pfb_init_noray] Writing to {xds_store.url}")

    ms_path = str(args.ms)
    msstore = DaskMSStore(ms_path.rstrip("/"))
    mslist = msstore.fs.glob(ms_path.rstrip("/"))
    ms = list(map(msstore.fs.unstrip_protocol, mslist))

    print(f"[pfb_init_noray] Constructing mappings for {ms}")
    (
        row_mapping, freq_mapping, time_mapping,
        freqs, utimes, ms_chunks, gains, radecs,
        chan_widths, uv_max, antpos, poltype,
    ) = construct_mappings(ms, None)

    group_by = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]
    columns = ("DATA", "UVW", "ANTENNA1", "ANTENNA2", "TIME", "INTERVAL", "FLAG_ROW", "FLAG")
    schema = {"DATA": {"dims": ("chan", "corr")}, "FLAG": {"dims": ("chan", "corr")}}

    # Build band mapping
    freq_groups = []
    freq_sgroups = []
    sgroup = 0
    for ms_name in ms:
        for idt, freq in freqs[ms_name].items():
            if not len(freq_groups):
                freq_groups.append(freq)
                freq_sgroups.append(sgroup)
                sgroup += freq_mapping[ms_name][idt]["counts"].size
            else:
                in_group = any(freq.size == fs.size and np.all(freq == fs) for fs in freq_groups)
                if not in_group:
                    freq_groups.append(freq)
                    freq_sgroups.append(sgroup)
                    sgroup += freq_mapping[ms_name][idt]["counts"].size

    msddid2bid = {}
    for ms_name in ms:
        msddid2bid[ms_name] = {}
        for idt, freq in freqs[ms_name].items():
            for sgroup, fs in zip(freq_sgroups, freq_groups):
                if freq.size == fs.size and np.all(freq == fs):
                    msddid2bid[ms_name][idt] = sgroup

    t0 = time.time()
    ntasks = 0
    times_out = []
    freqs_out = []

    for ims, ms_name in enumerate(ms):
        xds = xds_from_ms(ms_name, columns=columns, table_schema=schema, group_cols=group_by)
        for ds in xds:
            fid = ds.FIELD_ID
            ddid = ds.DATA_DESC_ID
            scanid = ds.SCAN_NUMBER
            idt = f"FIELD{fid}_DDID{ddid}_SCAN{scanid}"

            titr = enumerate(zip(
                time_mapping[ms_name][idt]["start_indices"],
                time_mapping[ms_name][idt]["counts"],
            ))
            for ti, (tlow, tcounts) in titr:
                t_index = slice(tlow, tlow + tcounts)
                ridx = row_mapping[ms_name][idt]["start_indices"][t_index]
                rcnts = row_mapping[ms_name][idt]["counts"][t_index]
                row_index = slice(ridx[0], ridx[-1] + rcnts[-1])

                fitr = enumerate(zip(
                    freq_mapping[ms_name][idt]["start_indices"],
                    freq_mapping[ms_name][idt]["counts"],
                ))
                b0 = msddid2bid[ms_name][idt]
                for fi, (flow, fcounts) in fitr:
                    nu_index = slice(flow, flow + fcounts)
                    subds = ds[{"row": row_index, "chan": nu_index}]

                    result = stokes_vis(
                        dc1="DATA", dc2=None, operator=None,
                        ds=subds, jones=None,
                        freq=freqs[ms_name][idt][nu_index],
                        chan_width=chan_widths[ms_name][idt][nu_index],
                        utime=utimes[ms_name][idt][t_index],
                        tbin_idx=ridx, tbin_counts=rcnts,
                        chan_low=flow, chan_high=flow + fcounts,
                        radec=radecs[ms_name][idt],
                        antpos=antpos[ms_name],
                        poltype=poltype[ms_name],
                        xds_store=xds_store.url,
                        bandid=b0 + fi, timeid=ti, msid=ims,
                        precision=args.precision,
                        sigma_column=None,
                        weight_column=None,
                        product=args.product,
                        check_ants=False,
                        chan_average=1,
                        bda_decorr=1.0,
                        max_field_of_view=3.0,
                        beam_model=None,
                        wgt_mode="l2",
                    )
                    if result is not None:
                        times_out.append(result[0])
                        freqs_out.append(result[1])
                    ntasks += 1
                    print(f"[pfb_init_noray] Band {b0+fi} time {ti} done ({ntasks} total)", flush=True)

    print(f"[pfb_init_noray] All {ntasks} chunks done in {time.time()-t0:.1f}s")
    print(f"[pfb_init_noray] {len(np.unique(freqs_out))} bands, {len(np.unique(times_out))} times")
    print(f"[pfb_init_noray] Output: {xds_store.url}")


if __name__ == "__main__":
    main()
