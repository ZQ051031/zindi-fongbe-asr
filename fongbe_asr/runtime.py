import ctypes
import errno
import os


def one_cpu():
    for key in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})


def deny_network():
    """Install a Linux seccomp filter before importing inference dependencies."""
    lib = ctypes.CDLL("libseccomp.so.2")
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None
    allow = 0x7FFF0000
    reject = 0x00050000 | errno.EPERM
    names = (b"socket", b"socketpair", b"socketcall", b"connect", b"bind", b"listen",
             b"accept", b"accept4", b"sendto", b"sendmsg", b"sendmmsg", b"io_uring_setup")
    context = lib.seccomp_init(allow)
    if not context:
        raise OSError("seccomp_init failed")
    try:
        for name in names:
            number = lib.seccomp_syscall_resolve_name(name)
            if number >= 0 and lib.seccomp_rule_add(context, reject, number, 0) != 0:
                raise OSError(f"Cannot block syscall {name!r}")
        if lib.seccomp_load(context) != 0:
            raise OSError("Cannot install offline benchmark filter")
    finally:
        lib.seccomp_release(context)
