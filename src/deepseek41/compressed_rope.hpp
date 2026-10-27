#pragma once
namespace strata::deepseek41::detail {
// Internal fixed-1017 RoPE, sharing the frequency table with compressed KV and
// index keys/queries. BF16-rounded F32 rows [heads,512], adjacent final 64 values.
// Caller validates position, buffers, stream and rounds inputs. Asynchronous.
void rotate_compressed(float* values, int heads, int position, bool inverse, void* stream);
}
