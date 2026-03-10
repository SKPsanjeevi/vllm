# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm import envs
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.v1.attention.backend import AttentionCGSupport, AttentionLayer, MultipleOf
from vllm.v1.kv_cache_interface import AttentionSpec


class AiterMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA"

    @staticmethod
    def get_impl_cls() -> type["AiterMLAImpl"]:
        return AiterMLAImpl

    @staticmethod
    def get_builder_cls() -> type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder


@dataclass
class AiterMLADecodeMetadata(MLACommonDecodeMetadata):
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: torch.Tensor | None = None
    # The page indices of the paged kv cache
    paged_kv_indices: torch.Tensor | None = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: torch.Tensor | None = None
    # The query indptr, shape : [num_decode + 1]
    qo_indptr: torch.Tensor | None = None
    # The dtype of MLA out tensor
    attn_out_dtype: torch.dtype = torch.bfloat16
    # The max query output length: int
    max_qo_len: int | None = None

    work_meta_data: torch.Tensor | None = None
    work_indptr: torch.Tensor | None = None
    work_info_set: torch.Tensor | None = None
    reduce_indptr: torch.Tensor | None = None
    reduce_final_map: torch.Tensor | None = None
    reduce_partial_map: torch.Tensor | None = None


class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    pass


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):
    # TODO(luka, lucas): audit this as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, AiterMLAMetadata
        )

        self.compilation_config = vllm_config.compilation_config
        self.decode_attn_out_dtype = vllm_config.model_config.dtype
        # kernel block size is always 1.
        max_num_pages_per_req = vllm_config.model_config.max_model_len
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req

        # Preparing persistent buffers
        # TODO: we can disambiguate between decode and mixed-prefill decode here
        # so we can only use the persistent buffer if a cudagraph is actually
        # being used.

        # paged_kv_last_page_len is always 1s (kernel block size is always 1),
        # so we create it once and reuse slices in both eager and cudagraph modes.
        self.paged_kv_last_page_len = torch.ones(
            max_num_reqs, dtype=torch.int32, device=device
        )

        if envs.VLLM_ROCM_USE_AITER_MLA_PERSISTENT:
            from aiter import dtypes, get_mla_metadata_info_v1

            num_attention_heads = vllm_config.model_config.get_num_attention_heads(
                vllm_config.parallel_config
            )
            q_dtype = self.decode_attn_out_dtype
            kv_cache_dtype_str = getattr(
                vllm_config.cache_config, "cache_dtype", "auto"
            )
            if kv_cache_dtype_str in ("fp8", "fp8_e4m3", "fp8_e5m2"):
                kv_cache_dtype_str = "fp8"
            else:
                kv_cache_dtype_str = "bf16"
            kv_dtype = dtypes.d_dtypes.get(kv_cache_dtype_str, dtypes.bf16)
            (
                (work_meta_data_size, work_meta_data_type),
                (work_indptr_size, work_indptr_type),
                (work_info_set_size, work_info_set_type),
                (reduce_indptr_size, reduce_indptr_type),
                (reduce_final_map_size, reduce_final_map_type),
                (reduce_partial_map_size, reduce_partial_map_type),
            ) = get_mla_metadata_info_v1(
                max_num_reqs,
                1,
                num_attention_heads,
                q_dtype,
                kv_dtype,
                is_sparse=False,
                fast_mode=True,
            )
            self._mla_work_meta_data = torch.empty(
                work_meta_data_size, dtype=work_meta_data_type, device=device
            )
            self._mla_work_indptr = torch.empty(
                work_indptr_size, dtype=work_indptr_type, device=device
            )
            self._mla_work_info_set = torch.empty(
                work_info_set_size, dtype=work_info_set_type, device=device
            )
            self._mla_reduce_indptr = torch.empty(
                reduce_indptr_size, dtype=reduce_indptr_type, device=device
            )
            self._mla_reduce_final_map = torch.empty(
                reduce_final_map_size, dtype=reduce_final_map_type, device=device
            )
            self._mla_reduce_partial_map = torch.empty(
                reduce_partial_map_size,
                dtype=reduce_partial_map_type,
                device=device,
            )
            self._num_attention_heads = num_attention_heads

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )
            self.paged_kv_indices = torch.zeros(
                max_num_pages, dtype=torch.int32, device=device
            )

            self.qo_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> AiterMLADecodeMetadata:
        # kernel block size is always 1, although the kv block size is not 1.
        device = self.device
        num_reqs = seq_lens_device.size(0)

        mask = torch.arange(
            block_table_tensor.size(1), dtype=block_table_tensor.dtype, device=device
        ).unsqueeze(0) < seq_lens_device.unsqueeze(1)
        paged_kv_indices = block_table_tensor[mask]

        # kernel block size is always 1, so each page has exactly 1 token.
        # last_page_len is always 1 - just slice the pre-initialized buffer.
        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]

        paged_kv_indptr = torch.cat(
            [
                torch.zeros(1, dtype=seq_lens_device.dtype, device=device),
                seq_lens_device.cumsum(dim=0, dtype=torch.int32),
            ]
        )
        qo_len = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_qo_len = qo_len.max().item()

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            num_actual_pages = paged_kv_indices.size(0)

            self.paged_kv_indices[:num_actual_pages].copy_(
                paged_kv_indices, non_blocking=True
            )
            self.paged_kv_indices[num_actual_pages:].fill_(-1)
            paged_kv_indices = self.paged_kv_indices[:num_actual_pages]

            self.paged_kv_indptr[: 1 + num_reqs].copy_(
                paged_kv_indptr, non_blocking=True
            )
            self.paged_kv_indptr[1 + num_reqs :].fill_(paged_kv_indptr[-1])
            paged_kv_indptr = self.paged_kv_indptr[: 1 + num_reqs]

            # paged_kv_last_page_len already uses the pre-initialized buffer slice
            # (set above), so no copy needed - buffer is always 1s.

            self.qo_indptr[: 1 + num_reqs].copy_(
                query_start_loc_device, non_blocking=True
            )
            self.qo_indptr[1 + num_reqs :] = query_start_loc_device[-1]
            qo_indptr = self.qo_indptr[: 1 + num_reqs]

        else:
            qo_indptr = torch.arange(
                0, num_reqs + 1, step=1, dtype=torch.int32, device=device
            )

        decode_work_meta_data = None
        decode_work_indptr = None
        decode_work_info_set = None
        decode_reduce_indptr = None
        decode_reduce_final_map = None
        decode_reduce_partial_map = None
        if getattr(self, "_mla_work_meta_data", None) is not None:
            from aiter import get_mla_metadata_v1

            get_mla_metadata_v1(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_last_page_len,
                self._num_attention_heads,
                1,
                True,
                self._mla_work_meta_data,
                self._mla_work_info_set,
                self._mla_work_indptr,
                self._mla_reduce_indptr,
                self._mla_reduce_final_map,
                self._mla_reduce_partial_map,
                page_size=1,
                kv_granularity=16,
                max_seqlen_qo=max_qo_len,
                uni_seqlen_qo=max_qo_len,
                fast_mode=True,
            )
            decode_work_meta_data = self._mla_work_meta_data
            decode_work_indptr = self._mla_work_indptr
            decode_work_info_set = self._mla_work_info_set
            decode_reduce_indptr = self._mla_reduce_indptr
            decode_reduce_final_map = self._mla_reduce_final_map
            decode_reduce_partial_map = self._mla_reduce_partial_map

        attn_metadata = AiterMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            qo_indptr=qo_indptr,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            max_qo_len=max_qo_len,
            attn_out_dtype=self.decode_attn_out_dtype,
            work_meta_data=decode_work_meta_data,
            work_indptr=decode_work_indptr,
            work_info_set=decode_work_info_set,
            reduce_indptr=decode_reduce_indptr,
            reduce_final_map=decode_reduce_final_map,
            reduce_partial_map=decode_reduce_partial_map,
        )

        return attn_metadata


class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )
        _valid_heads = num_heads in (4, 8) or (
            num_heads % 16 == 0 and 16 <= num_heads <= 128
        )
        assert _valid_heads, (
            f"Aiter MLA supports num_heads of 4, 8, or multiples of 16 "
            f"in [16, 128].\n"
            f"Provided {num_heads} number of heads.\n"
            "Try adjusting tensor_parallel_size value."
        )
        self._needs_head_repeat = num_heads < 16
        self._head_repeat_factor = 16 // num_heads if num_heads < 16 else 1
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "Aiter MLA does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        from aiter import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        output = self.flash_attn_varlen_func(  # type: ignore[call-arg]
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            return_lse=return_softmax_lse,
            **kwargs,
        )

        return output

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None
        assert attn_metadata.decode.max_qo_len is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]

        if self._needs_head_repeat:
            q = q.repeat_interleave(self._head_repeat_factor, dim=1)
            kernel_num_heads = 16
        else:
            kernel_num_heads = self.num_heads

        o = torch.zeros(
            B,
            kernel_num_heads,
            self.kv_lora_rank,
            dtype=attn_metadata.decode.attn_out_dtype,
            device=q.device,
        )

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        decode = attn_metadata.decode
        rocm_aiter_ops.mla_decode_fwd(
            q,
            kv_buffer,
            o,
            self.scale,
            decode.qo_indptr,
            decode.max_qo_len,
            decode.paged_kv_indptr,
            decode.paged_kv_indices,
            decode.paged_kv_last_page_len,
            q_scale=layer._q_scale,
            kv_scale=layer._k_scale,
            work_meta_data=decode.work_meta_data,
            work_indptr=decode.work_indptr,
            work_info_set=decode.work_info_set,
            reduce_indptr=decode.reduce_indptr,
            reduce_final_map=decode.reduce_final_map,
            reduce_partial_map=decode.reduce_partial_map,
        )

        if self._needs_head_repeat:
            o = o[:, :: self._head_repeat_factor, :]

        return o, None
