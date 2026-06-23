# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  📂 mmap Allocator —— 基于 mmap 的 CPU 内存分配，用于 HiCache Host KV Pool            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

import ctypes
import ctypes.util
import logging
import math
import mmap
import os
import weakref

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Load libc once at module level so munmap is callable safely at GC/shutdown time.
# Resolve the SONAME via find_library so the allocator also works on systems
# whose libc is not named "libc.so.6" (e.g. musl / Alpine).
try:
    _libc_name = ctypes.util.find_library("c") or "libc.so.6"
    _libc = ctypes.CDLL(_libc_name, use_errno=True)
    _libc.mmap.restype = ctypes.c_void_p
    _libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    _libc.munmap.restype = ctypes.c_int
    _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
except OSError:
    _libc = None

# MAP_POPULATE: 立即预填充页表项，强制内核在 mmap 返回前分配并映射所有物理页面。
#   - 消除后续访问时的缺页中断（page fault）
#   - 确保 cudaHostRegister 锁定时物理页面已存在，避免 GPU 读到脏数据
#   - Python 3.11+ 的 mmap 模块才暴露该常量，低版本需要硬编码值 0x08000
_MAP_POPULATE = getattr(mmap, "MAP_POPULATE", 0x08000)

# MAP_HUGETLB: 启用 HugeTLB 大页分配。
#   大页能显著减少 TLB（Translation Lookaside Buffer）miss，
#   对于 KV cache 这种几十 GB 的连续内存池至关重要。
#   - 4KB 页: 100GB 需要 ~2600万个 TLB 条目（远超 CPU TLB 容量）
#   - 2MB 页: 100GB 需要 ~51200 个条目
#   - 1GB 页: 100GB 只需 ~100 个条目
# MAP_HUGETLB 和 MAP_HUGE_* 是 Linux 专有标志，Python mmap 模块不暴露，
# 因此大页路径必须通过 ctypes 直接调用 libc mmap。
_MAP_HUGETLB = 0x40000

# MAP_HUGE_2MB / MAP_HUGE_1GB: 指定大页的具体大小。
#   编码规则: (log2(page_size)) << MAP_HUGE_SHIFT(26)
#   - 2MB  = 2^21 → 21 << 26 = 0x1400000
#   - 1GB  = 2^30 → 30 << 26 = 0x78000000
_MAP_HUGE_2MB = 21 << 26  # 0x1400000
_MAP_HUGE_1GB = 30 << 26  # 0x78000000

# MAP_FAILED: mmap 失败时的返回值，即 (void*)-1。
#   ctypes.c_void_p(-1).value 将其转为 Python 可比较的整数值。
_MAP_FAILED = ctypes.c_void_p(-1).value


def _alloc_hugepage(n_bytes: int, alloc_bytes: int, extra_flags: int) -> ctypes.Array:
    """通过 libc mmap 分配大页内存，返回持有内存的 ctypes 数组。

    为什么不能直接用 Python 的 mmap.mmap 分配大页？
    Python 的 mmap 模块不暴露 MAP_HUGETLB 等大页标志位（这些是 Linux 专有
    且不在 POSIX 标准中），因此需要用 ctypes 直接调用 libc 的 mmap。

    Args:
        n_bytes:    实际需要的字节数（未对齐），用于创建 ctypes 数组。
        alloc_bytes: 对齐到页大小后的分配字节数，作为 mmap 的 length 参数。
        extra_flags: 大页标志位（MAP_HUGETLB | MAP_HUGE_2MB 或 MAP_HUGE_1GB）。

    Returns:
        ctypes.c_uint8 数组，大小为 n_bytes，底层映射了 alloc_bytes 的物理内存。

    生命周期：
        weakref.finalize 在 array 被 GC 时自动调用 munmap 释放物理内存。
        torch.frombuffer(array, ...) 创建的 tensor 会持有 array 引用，
        因此 tensor 释放时 array 才被回收，随后自动 munmap。
    """
    # 直接调用 libc mmap，传入大页标志位
    # 参数含义（按顺序）：
    #   addr=NULL     → 由内核选择映射地址
    #   length        → 映射长度（已页对齐）
    #   prot          → PROT_READ | PROT_WRITE（可读写）
    #   flags         → MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE | 大页标志
    #   fd=-1         → MAP_ANONYMOUS 模式下忽略（无文件后备）
    #   offset=0      → MAP_ANONYMOUS 模式下必须为 0
    ptr = _libc.mmap(
        None,
        alloc_bytes,
        mmap.PROT_READ | mmap.PROT_WRITE,
        mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | _MAP_POPULATE | extra_flags,
        -1,
        0,
    )
    # MAP_FAILED = (void*)-1，表示 mmap 失败
    if ptr is None or ptr == _MAP_FAILED:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))

    # 用 ctypes 数组包装映射的内存区域
    # from_address 不拥有内存，仅创建一个指向已映射地址的数组视图
    array = (ctypes.c_uint8 * n_bytes).from_address(ptr)

    # 注册析构回调：array 被 GC 回收时自动 munmap
    # 注意 munmap 需要传入 alloc_bytes 而非 n_bytes，因为实际映射的大小是 alloc_bytes
    weakref.finalize(array, _libc.munmap, ctypes.c_void_p(ptr), alloc_bytes)
    return array


def alloc_mmap(dims: tuple, dtype: torch.dtype) -> torch.Tensor:
    """通过匿名 mmap 分配主机端 tensor，可选大页支持。

    为什么不用 torch.empty(pin_memory=True)？
    1. torch.empty 使用惰性分配（demand paging），调用时只分配虚拟地址，
       物理页面在首次写入时才分配。若先 pin_memory 再写入，CUDA 驱动
       锁定的可能是尚未分配的页面，后续缺页会导致 GPU 读到脏数据。
    2. torch.empty 不暴露大页接口，无法使用 HugeTLB 减少 TLB miss。

    本函数的解决方案：
    - MAP_POPULATE: 立即预填充所有物理页面，消除缺页延迟和竞争。
    - MAP_SHARED: 防止 COW（Copy-on-Write），确保 fork 后页面不被替换。
    - 支持通过 SGLANG_HUGEPAGE_SIZE 环境变量启用 2MB 或 1GB 大页。

    Args:
        dims:  张量形状，如 (num_layers, num_heads, head_dim)。
        dtype: 数据类型，如 torch.float16。

    Returns:
        CPU tensor，由 mmap 匿名映射支持。释放 tensor 时自动 munmap。

    生命周期：
        普通路径: Python mmap.mmap 对象被 torch.frombuffer 内部持有，
                  tensor 释放 → mm.__del__() → munmap。
        大页路径: ctypes 数组被 torch.frombuffer 内部持有，
                  数组 GC → weakref.finalize → libc munmap。
    """
    # 每次调用都重新读取环境变量（不缓存），
    # 保证 envs.SGLANG_HUGEPAGE_SIZE.override() 在测试中能正常工作
    hugepage_size = (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip().upper()

    # 计算所需总字节数: 元素数 × 单元素字节数
    # torch.empty([], dtype=dtype).element_size() 避免了硬编码各类型的字节数
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()

    # ---- 解析大页配置 ----
    if hugepage_size == "":
        page_size, extra_flags = mmap.PAGESIZE, 0  # 默认 4KB 页
    elif hugepage_size == "2MB":
        # MAP_HUGETLB: 启用大页; MAP_HUGE_2MB: 指定 2MB 大页大小
        page_size, extra_flags = 2 * 1024 * 1024, _MAP_HUGETLB | _MAP_HUGE_2MB
    elif hugepage_size == "1GB":
        # MAP_HUGE_1GB: 指定 1GB 大页大小（需要 CPU 支持）
        page_size, extra_flags = 1024 * 1024 * 1024, _MAP_HUGETLB | _MAP_HUGE_1GB
    else:
        logger.warning(
            "Unrecognized SGLANG_HUGEPAGE_SIZE=%r; expected '2MB' or '1GB'. "
            "Falling back to plain page-size mmap.",
            envs.SGLANG_HUGEPAGE_SIZE.get(),
        )
        page_size, extra_flags = mmap.PAGESIZE, 0

    # 向上取整到页大小倍数，确保 mmap 映射区域页对齐
    alloc_bytes = math.ceil(n_bytes / page_size) * page_size

    # ---- 尝试大页分配路径 ----
    if extra_flags:
        if _libc is None:
            # libc 加载失败（例如非 Linux 系统），无法使用大页
            logger.error(
                "Hugepage mmap requested but libc.so.6 could not be loaded; "
                "falling back to plain mmap. SGLANG_HUGEPAGE_SIZE=%s will be ignored.",
                hugepage_size,
            )
        else:
            try:
                # 通过 libc 直接调用 mmap 并传入大页标志位
                array = _alloc_hugepage(n_bytes, alloc_bytes, extra_flags)
                # torch.frombuffer 将 ctypes 数组包装为 tensor，
                # 不复制数据，tensor 直接共享 mmap 的底层内存
                return torch.frombuffer(
                    array, dtype=dtype, count=math.prod(dims)
                ).reshape(dims)
            except OSError as e:
                # 大页分配失败常见原因：
                # - 系统未预留大页 (echo N > /proc/sys/vm/nr_hugepages)
                # - 预留的大页数量不足
                # - 用户没有大页使用权限
                logger.error(
                    "Hugepage mmap via libc failed (%s); falling back to plain mmap. "
                    "SGLANG_HUGEPAGE_SIZE=%s will be ignored.",
                    e,
                    hugepage_size,
                )
        # 回退：重新按普通 4KB 页对齐
        alloc_bytes = math.ceil(n_bytes / mmap.PAGESIZE) * mmap.PAGESIZE

    # ---- 普通 mmap 路径（默认路径或大页失败的回退路径） ----
    # mmap 参数：
    #   fd=-1         → 匿名映射（无文件后备）
    #   length        → 页对齐的分配大小
    #   flags:
    #     MAP_SHARED     → 共享映射，防止 COW
    #     MAP_ANONYMOUS  → 不与文件关联，等价于 malloc + 匿名内存
    #     MAP_POPULATE   → 立即填充页表，预分配物理页面
    #   prot:
    #     PROT_READ | PROT_WRITE → 可读写
    #
    # torch.frombuffer 持有 mm 对象的引用，因此 mm 的生命周期与 tensor 一致，
    # tensor 释放时 mm.__del__() 自动调用 munmap
    mm = mmap.mmap(
        -1,
        alloc_bytes,
        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | _MAP_POPULATE,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    return torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims)
