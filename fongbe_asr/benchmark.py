import json
import os
import resource
import socket
import time

from .infer import make_parser, parse_args, run
from .runtime import deny_network, one_cpu


LIMITS = dict(weights_bytes=1_000_000_000, rtf=1.0, peak_rss_bytes=2_000_000_000)


def main():
    started = time.perf_counter()
    args = parse_args(make_parser("Benchmark offline, single-CPU inference (Linux)"))
    one_cpu()
    if len(os.sched_getaffinity(0)) != 1:
        raise RuntimeError("Benchmark must be pinned to exactly one CPU")
    deny_network()
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        pass
    else:
        probe.close()
        raise RuntimeError("Offline socket probe unexpectedly succeeded")
    size = args.checkpoint.stat().st_size + (args.lm.stat().st_size if args.lm else 0)
    count, duration = run(args)
    elapsed = time.perf_counter() - started
    stats = dict(weights_bytes=size, rtf=elapsed / duration,
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    passed = all(stats[key] <= maximum for key, maximum in LIMITS.items())
    print(json.dumps(dict(stats, limits=LIMITS, passed=passed, audio_seconds=duration,
                          wall_seconds=elapsed, recordings=count, pinned_cpus=1,
                          network="blocked by Linux seccomp; AF_INET socket probe denied"), indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
