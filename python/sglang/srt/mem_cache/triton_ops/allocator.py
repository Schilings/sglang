import triton
import triton.language as tl


@triton.jit
def alloc_extend_kernel(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    """
    为一批请求的 extend 部分（prefill 新 token）分配 KV slot 索引。

    每个 program 处理一个请求，将该请求的 extend 部分分成三段填充 out_indices:
      Part 1: 填充旧 page 的剩余空间（不消耗新 page）
      Part 2: 填充完整的新 page（从 free_pages 取）
      Part 3: 填充新 page 的前半段（与 Part 2 最后一个 page 共享）

    Part 1 何时执行？
    ─────────────────
    当 prefix_len 不是 page_size 的倍数时，表明前缀的最后一个 page 还有空位，
    需要用 Part 1 填满。这取决于 prefix 的来源：

    • RadixCache 路径（有 prefix 匹配）：
      radix cache 存储时 key 经过了 .page_aligned(page_size) 截断，
      cache_unfinished_req 也只会缓存 page-aligned 长度的 KV indices。
      因此 match_prefix 返回的 prefix_len 一定是 page_size 的倍数 → Part 1 = 0。

    • ChunkCache / Streaming Session 路径（无 radix prefix 匹配）：
      chunked prefill 第二次调度时，init_next_round_input 调用时 tree_cache=None，
      不会重新 match prefix。prefix_indices 由上一轮 cache_unfinished_req 直接设置
      （取全部 kv_indices，不做 page-aligned 截断）。
      如果上一轮 extend 没填满一个 page，prefix_len 就不是 page_size 的倍数 → Part 1 > 0，
      将拼接到自己之前分配的半满 page 后面。

    首次 prefill（无历史 prefix）的 prefix_len = 0，也是 page_size 的倍数 → Part 1 = 0。
    """
    pid = tl.program_id(0)

    # 计算当前请求在 out_indices 中的起始位置:
    # output_start_loc = sum(前面所有请求的 extend_len)
    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    # 计算当前请求需要多少个新 page，以及在所有请求的新 page 中的起始偏移:
    # num_new_pages = ceil(seq_len/page_size) - ceil(pre_len/page_size)
    # 两侧都用 ceil，差值自动排除了"半满的旧 page"（它已在 pre_len 的 ceil 中被计入）
    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 1: 填充旧 page 的剩余空间
    # 当 pre_len 不对齐到 page_size 时，前缀最后一个 page 还有空位。
    # 例如 pre_len=50, page_size=16 → page 3 已用到 slot 49，剩余 [50-63] 共 6 个空位。
    # 这些 slot 已经分配给该请求（之前 alloc 时整页分配的），直接用 last_loc+1 接上即可。
    # 如果 pre_len 是 page_size 的倍数（如从 radix cache 匹配），则 num_part1=0。
    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: 填充完整的新 page
    # 每个 token 的 slot = page_number * page_size + offset_in_page
    # page_number 从 free_pages 中按顺序取。
    # 使用动态 blocked loop 而非完全展开，避免 Triton 对 extend 大小做 constexpr 展开导致
    # 多次编译 kernel。
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )
    BLOCK_EXTEND: tl.constexpr = 4096
    num_blocks = (num_part2 + BLOCK_EXTEND - 1) // BLOCK_EXTEND
    for block_id in range(num_blocks):
        offset_in_block = tl.arange(0, BLOCK_EXTEND)
        offset = block_id * BLOCK_EXTEND + offset_in_block
        mask = offset < num_part2
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset // page_size,
            mask=mask,
        )
        tl.store(
            out_indices + output_start_loc + num_part1 + offset,
            page_start * page_size + offset % page_size,
            mask=mask,
        )
    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: 填充新 page 的前半段
    # seq_len 不对齐时，最后一个 page 只填前几个 slot。
    # 这个 page 和 Part 2 的最后一个 page 可能是同一个（如果 Part 2 非空），
    # 也可能是独立的新 page（如果 Part 2 为空），都通过 free_pages 取。
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


@triton.jit
def alloc_decode_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    """
    为一批 decode 请求各分配 1 个 KV slot 索引。

    decode 每步只新增 1 个 token，所以 pre_len = seq_len - 1。
    两种情况:
      1. seq_len 没有跨越 page 边界 → 不需要新 page，直接用 last_loc + 1
      2. seq_len 跨越 page 边界 → 需要从 free_pages 取一个新 page，用其第一个 slot
    """
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    if num_page_start_loc_self == 0:
        # 当前 page 还有空位，直接接在 last_loc 后面
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        # 当前 page 已满，需要新 page 的第一个 slot
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)
