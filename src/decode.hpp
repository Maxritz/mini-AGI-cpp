#pragma once
#include "tensor.hpp"

namespace minagi {

struct DecodeWeights {
  double strength = 2.5;      // decoding.adapt_strength
  double decay = 0.88;        // decoding.adapt_decay
  int window = 64;            // adapt_window
  double rep_penalty = 1.0;   // decoding.rep_penalty
  int no_repeat_ngram = 0;
};

// Deterministic next token (port of minagi/decode.py pick_next, inference
// path). logits [1,V] FP32; prev_ids [1,L] (integral ids, most recent LAST).
// Returns the chosen id, or -1 when logits is not [1,V].
int pick_next(const mt::Tensor& logits, const mt::Tensor& prev_ids,
              const DecodeWeights& w);

}  // namespace minagi