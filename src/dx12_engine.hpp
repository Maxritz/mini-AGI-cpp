// src/dx12_engine.hpp
// Dx12ComputeEngine: wraps Dx12Context, manages PSOs for each shader entry point,
// and provides UploadTensor/ReadbackTensor helpers for mt::Tensor <-> GPU buffer.
// Guarded by #ifdef _WIN32.

#pragma once

#ifdef _WIN32

#include "dx12_context.hpp"
#include "tensor.hpp"
#include <d3d12.h>
#include <cstdint>
#include <string>
#include <vector>

namespace dx12 {

// Pipeline state objects for the dense shader entry points.
// Matches the 12-binding DSL in dense.hlsl / pool.hlsl.
class Dx12ComputeEngine {
public:
    Dx12ComputeEngine();
    ~Dx12ComputeEngine();

    // Initialize device + compile + create PSOs. Returns false on failure.
    bool init(const std::string& shader_dir);

    bool ready() const { return ready_; }

    ID3D12Device* device() const { return ctx_.device(); }
    ID3D12CommandQueue* queue() const { return ctx_.queue(); }
    ID3D12GraphicsCommandList* cmd_list() const { return ctx_.cmd_list(); }
    ID3D12RootSignature* root_sig() const { return ctx_.root_sig(); }

    // ---- Buffer management ----
    // Create a GPU buffer of the given size (in bytes). is_uav=true for output.
    Dx12Buffer create_buffer(uint64_t byte_size, bool is_uav);

    // Upload tensor data to a GPU buffer.
    void upload_buffer(const Dx12Buffer& buf, const mt::Tensor& tensor);

    // Readback: copy GPU buffer to a new mt::Tensor (fp32).
    mt::Tensor readback_buffer(const Dx12Buffer& buf, mt::Shape shape, mt::DType dtype);

    // ---- Command list management ----
    void reset();        // reset allocator + command list
    void close();        // close command list for submission
    void dispatch(uint32_t x, uint32_t y, uint32_t z);  // issue Dispatch
    void flush();        // close, execute, signal, wait
    void signal_fence();
    void wait_for_fence();

    // ---- Root constant / resource binding ----
    // Bind a buffer to a root descriptor range (binding slot 0-based, used with
    // SetComputeRootUnorderedAccessView / SetComputeRootShaderResourceView).
    void set_uav(UINT slot, const Dx12Buffer& buf);
    void set_srv(UINT slot, const Dx12Buffer& buf);

    // Push 32-bit constants into root param 1.
    void push_constants(const void* data, uint32_t num_32bit_values);

    // Set a PSO for a specific shader entry point.
    // Returns the PSO index.
    int load_compute_pso(const std::string& entry_point);

    // Bind PSO by index or name.
    void set_pso(int pso_index);
    void set_pso(const std::string& name);

private:
    Dx12Context ctx_;
    bool ready_ = false;

    struct PsoEntry {
        std::string name;
        ComPtr<ID3D12PipelineState> pso;
    };
    std::vector<PsoEntry> psos_;

    std::string shader_dir_;
};

} // namespace dx12

#endif // _WIN32
