"""FlashAttention 1 Algorithm 1 forward kernel exercise."""

import triton
import triton.language as tl

# Set to True only after implementing every kernel TODO below.
FORWARD_IMPLEMENTED = False


@triton.jit
def _flash_attention_forward(
    Q_ptr: tl.tensor,
    K_ptr: tl.tensor,
    V_ptr: tl.tensor,
    O_ptr: tl.tensor,
    M_ptr: tl.tensor,
    L_ptr: tl.tensor,
    softmax_scale: tl.tensor,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    O_STRIDES: tl.constexpr,
    STATE_STRIDES: tl.constexpr,
    QUERY_LENGTH: tl.constexpr,
    KEY_LENGTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CAUSAL: tl.constexpr,
) -> None:
    """Fill attention output using one program per batch/head pair.

    Args:
        Q_ptr (tl.tensor): FP16/BF16 query pointer, logical shape [B, Nq, H, D].
        K_ptr (tl.tensor): FP16/BF16 key pointer, logical shape [B, Nk, H, D].
        V_ptr (tl.tensor): FP16/BF16 value pointer, logical shape [B, Nk, H, D].
        O_ptr (tl.tensor): Contiguous FP32 output/state [B, Nq, H, D], initially zero.
        M_ptr (tl.tensor): Contiguous FP32 row maxima [B, Nq, H], initially -inf.
        L_ptr (tl.tensor): Contiguous FP32 row sums [B, Nq, H], initially zero.
        softmax_scale (tl.tensor): Positive finite score multiplier.
        Q_STRIDES (tl.constexpr): Query element strides in [B, Nq, H, D] order.
        K_STRIDES (tl.constexpr): Key element strides in [B, Nk, H, D] order.
        V_STRIDES (tl.constexpr): Value element strides in [B, Nk, H, D] order.
        O_STRIDES (tl.constexpr): Output element strides in [B, Nq, H, D] order.
        STATE_STRIDES (tl.constexpr): Shared M/L element strides in [B, Nq, H] order.
        QUERY_LENGTH (tl.constexpr): Number of query rows per head.
        KEY_LENGTH (tl.constexpr): Number of key/value rows per head.
        HEAD_DIM (tl.constexpr): Per-head width: 32, 64, or 128.
        BLOCK_Q (tl.constexpr): Query tile height.
        BLOCK_K (tl.constexpr): Key/value tile height.
        CAUSAL (tl.constexpr): Whether to apply bottom-right causal masking.

    Returns:
        None: Once implemented, write normalized attention to O and final statistics to M/L.

    Note:
        Grid axis 0 selects the batch and axis 1 selects the head. Each program
        owns all query rows of that pair. O/M/L remain in HBM across tile updates.
        The launcher converts the final FP32 O to the input dtype.
    """
    # TODO: Identify this program's batch/head and construct tile offsets and bounds masks.
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    k_block_offsets = tl.arange(0, BLOCK_K)
    q_block_offsets = tl.arange(0, BLOCK_Q)
    head_dim_offsets = tl.arange(0, HEAD_DIM)

    # TODO: Traverse K/V tiles in the outer loop and load the current K/V tile.
    for k_block_start_offset in range(0, KEY_LENGTH, BLOCK_K):
        k_tile = tl.load(
            K_ptr
            + batch_idx * K_STRIDES[0]
            + (k_block_start_offset + k_block_offsets[:, None]) * K_STRIDES[1]
            + head_idx * K_STRIDES[2]
            + head_dim_offsets[None, :] * K_STRIDES[3],
            mask=(k_block_start_offset + k_block_offsets < KEY_LENGTH)[:, None],
            other=0.0,
        )
        kT_tile = tl.trans(k_tile)
        v_tile = tl.load(
            V_ptr
            + batch_idx * V_STRIDES[0]
            + (k_block_start_offset + k_block_offsets[:, None]) * V_STRIDES[1]
            + head_idx * V_STRIDES[2]
            + head_dim_offsets[None, :] * V_STRIDES[3],
            mask=(k_block_start_offset + k_block_offsets < KEY_LENGTH)[:, None],
            other=0.0,
        )

        # TODO: Traverse Q tiles in the inner loop and load Q plus the current O/M/L state.
        for q_block_start_offset in range(0, QUERY_LENGTH, BLOCK_Q):
            last_q_idx = tl.minimum(q_block_start_offset + BLOCK_Q, QUERY_LENGTH) - 1
            if QUERY_LENGTH - last_q_idx > KEY_LENGTH - k_block_start_offset and CAUSAL:
                # Skip Q tiles that are fully masked by the current K/V tile.
                pass
            else:
                q_tile = tl.load(
                    Q_ptr
                    + batch_idx * Q_STRIDES[0]
                    + (q_block_start_offset + q_block_offsets[:, None]) * Q_STRIDES[1]
                    + head_idx * Q_STRIDES[2]
                    + head_dim_offsets[None, :] * Q_STRIDES[3],
                    mask=(q_block_start_offset + q_block_offsets < QUERY_LENGTH)[:, None],
                    other=0.0,
                )

                q_kT_tile = tl.dot(q_tile, kT_tile) * softmax_scale
                causal_mask = (
                    (
                        (QUERY_LENGTH - q_block_start_offset - q_block_offsets)[:, None]
                        <= (KEY_LENGTH - k_block_start_offset - k_block_offsets)[None, :]
                    )
                    if CAUSAL
                    else tl.full((BLOCK_Q, BLOCK_K), True, dtype=tl.int1)
                )
                k_mask = k_block_start_offset + k_block_offsets[None, :] < KEY_LENGTH
                q_mask = q_block_start_offset + q_block_offsets[:, None] < QUERY_LENGTH
                mask = causal_mask & k_mask & q_mask
                q_kT_tile = tl.where(mask, q_kT_tile, float("-inf"))

                o_tile = tl.load(
                    O_ptr
                    + batch_idx * O_STRIDES[0]
                    + (q_block_start_offset + q_block_offsets[:, None]) * O_STRIDES[1]
                    + head_idx * O_STRIDES[2]
                    + head_dim_offsets[None, :] * O_STRIDES[3],
                    mask=(q_block_start_offset + q_block_offsets < QUERY_LENGTH)[:, None],
                    other=0.0,
                )
                m_tile = tl.load(
                    M_ptr
                    + batch_idx * STATE_STRIDES[0]
                    + (q_block_start_offset + q_block_offsets) * STATE_STRIDES[1]
                    + head_idx * STATE_STRIDES[2],
                    mask=q_block_start_offset + q_block_offsets < QUERY_LENGTH,
                    other=float("-inf"),
                )
                l_tile = tl.load(
                    L_ptr
                    + batch_idx * STATE_STRIDES[0]
                    + (q_block_start_offset + q_block_offsets) * STATE_STRIDES[1]
                    + head_idx * STATE_STRIDES[2],
                    mask=q_block_start_offset + q_block_offsets < QUERY_LENGTH,
                    other=0.0,
                )

                q_kT_tile_max = tl.max(q_kT_tile, axis=1)
                new_m_tile = tl.maximum(m_tile, q_kT_tile_max)
                alpha = tl.exp(m_tile - new_m_tile)
                P = tl.exp(q_kT_tile - new_m_tile[:, None])
                new_l_tile = l_tile * alpha + tl.sum(P, axis=1)
                new_o_tile = (
                    o_tile * l_tile[:, None] * alpha[:, None] + tl.dot(P, v_tile.to(tl.float32))
                ) / new_l_tile[:, None]

                # Store the updated O/M/L state for valid rows before advancing to the next tile.
                tl.store(
                    O_ptr
                    + batch_idx * O_STRIDES[0]
                    + (q_block_start_offset + q_block_offsets[:, None]) * O_STRIDES[1]
                    + head_idx * O_STRIDES[2]
                    + head_dim_offsets[None, :] * O_STRIDES[3],
                    new_o_tile,
                    mask=(q_block_start_offset + q_block_offsets < QUERY_LENGTH)[:, None],
                )
                tl.store(
                    M_ptr
                    + batch_idx * STATE_STRIDES[0]
                    + (q_block_start_offset + q_block_offsets) * STATE_STRIDES[1]
                    + head_idx * STATE_STRIDES[2],
                    new_m_tile,
                    mask=q_block_start_offset + q_block_offsets < QUERY_LENGTH,
                )
                tl.store(
                    L_ptr
                    + batch_idx * STATE_STRIDES[0]
                    + (q_block_start_offset + q_block_offsets) * STATE_STRIDES[1]
                    + head_idx * STATE_STRIDES[2],
                    new_l_tile,
                    mask=q_block_start_offset + q_block_offsets < QUERY_LENGTH,
                )
