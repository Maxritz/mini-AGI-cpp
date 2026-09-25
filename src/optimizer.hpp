#pragma once
// src/optimizer.hpp
// AdamW optimizer for mini-AGI training.
//
// Maintains first/second moment estimates (exp_avg, exp_avg_sq) and step_count
// per-parameter. Moment tensors are named via the *_m/*_v convention used by
// store::save_paged (see is_moment_key in paged.cpp:
//   key ends with "_m" or "_v", or "|m" / "|v" / "|t").
//
// The step() method applies:
//   1. Global gradient clipping (if clip > 0)
//   2. AdamW weight decay (decoupled)
//   3. Bias-corrected first/second moment update
//   4. Parameter update: p = p - lr * m_hat / (sqrt(v_hat) + eps)

#include "tensor.hpp"
#include "store.hpp"   // store::is_moment_key, store::is_optim
#include "paged.hpp"   // minagi::paged::PagedPool::is_moment_key
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace minagi {

class AdamW {
public:
    AdamW(float lr = 1e-3f, float weight_decay = 0.01f,
          float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, float clip = 1.0f,
          int64_t step_count = 0);

    // Step the optimizer with a map of parameter name -> (param, grad).
    // The param tensor is modified in-place; moments are created on first sight.
    void step(const std::vector<std::pair<std::string, mt::Tensor*>>& params_grads);

    // Step with flat arrays for more control.
    void step(const std::vector<std::pair<std::string, std::pair<mt::Tensor*, mt::Tensor*>>>& params_grads);

    // Global gradient clipping: scales all grads by clip_norm / global_norm.
    void clip_gradients(const std::vector<std::pair<std::string, mt::Tensor*>>& params_grads);
    void clip_gradients(const std::vector<std::pair<std::string, std::pair<mt::Tensor*, mt::Tensor*>>>& params_grads);

    // Accessors for moment tensors (for save/load).
    // Returns tensor named "<key>_m" and "<key>_v".
    mt::Tensor* get_moment_m(const std::string& key);
    mt::Tensor* get_moment_v(const std::string& key);
    int64_t step_count() const { return step_count_; }

    // Build state dict with moments included (for store::save / save_paged).
    // Moment keys follow the store convention: "<param>_m", "<param>_v"
    // (e.g., "tok_emb.weight_m", "tok_emb.weight_v").
    // Returns a flat vector of (name, tensor) pairs.
    std::vector<std::pair<std::string, mt::Tensor>> state_dict() const;

    // Load state dict: populates moments from entries matching *_m/*_v.
    // Returns the number of moment pairs loaded.
    int load_state_dict(const std::vector<std::pair<std::string, mt::Tensor>>& sd);

    // Set parameters
    void set_lr(float lr) { lr_ = lr; }
    void set_weight_decay(float wd) { weight_decay_ = wd; }
    void set_betas(float b1, float b2) { beta1_ = b1; beta2_ = b2; }
    void set_eps(float e) { eps_ = e; }
    void set_clip(float c) { clip_ = c; }

    float lr() const { return lr_; }
    float weight_decay() const { return weight_decay_; }
    float beta1() const { return beta1_; }
    float beta2() const { return beta2_; }
    float eps() const { return eps_; }
    float clip() const { return clip_; }

private:
    float lr_;
    float weight_decay_;
    float beta1_;
    float beta2_;
    float eps_;
    float clip_;
    int64_t step_count_;
    int64_t t_;  // per-optimizer step counter (starts at 0, incremented each step())

    // Moment storage: key -> {exp_avg, exp_avg_sq}
    // Keys are the full parameter names (e.g., "tok_emb.weight").
    struct MomentState {
        mt::Tensor m;  // first moment (exp_avg)
        mt::Tensor v;   // second moment (exp_avg_sq)
    };
    std::unordered_map<std::string, MomentState> moments_;

public:
    // Internal: update a single parameter using its gradient.
    void update_param(mt::Tensor& param, mt::Tensor& grad,
                      const std::string& key);
};

} // namespace minagi
