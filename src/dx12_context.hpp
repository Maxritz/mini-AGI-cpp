// src/dx12_context.hpp
// Core DirectX 12 infrastructure: device, command queue, allocator, command list,
// fence synchronization, descriptor heaps, and root signature for the 12-binding DSL.
// Guarded by #ifdef _WIN32 — excluded on non-Windows platforms.

#pragma once

#ifdef _WIN32

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cstdint>
#include <string>
#include <vector>

namespace dx12 {

using Microsoft::WRL::ComPtr;

// Root signature descriptor ranges for the 12-binding DSL.
// Each binding is either an SRV (read) or UAV (read/write).
// Root parameter 0 = 12 descriptor ranges (one per binding, interleaved SRV/UAV).
// Root parameter 1 = 32-bit root constants (push constants).
struct RootSigLayout {
    // Descriptor range for binding `b`: UAV if is_uav[b], else SRV.
    static constexpr UINT kNumBindings = 12;
    static constexpr UINT kRootParamCount = 2;  // descriptor tables + push constants
};

// A GPU buffer with upload-default heap pattern:
// - upload_heap: CPU-writable staging buffer, CPU-accessible
// - default_heap: GPU-local resource, GPU-accessible for compute
// - gpu_address: D3D12_GPU_VIRTUAL_ADDRESS of the default heap resource
// - cpu_address: CPU virtual address of the upload heap (for writes)
struct Dx12Buffer {
    ComPtr<ID3D12Resource> upload_heap;   // staging (CPU-write, then CopyBufferRegion)
    ComPtr<ID3D12Resource> default_heap;  // GPU-local
    D3D12_GPU_VIRTUAL_ADDRESS gpu_address = 0;
    void* cpu_address = nullptr;          // mapped upload heap pointer
    uint64_t size = 0;
    bool is_uav = false;                  // UAV vs SRV
};

// Core DX12 device context: factory, device, queue, allocator, command list, fence.
class Dx12Context {
public:
    Dx12Context();
    ~Dx12Context();

    // Initialize DXGI factory, create D3D12 device (or return false on failure).
    bool init();

    ID3D12Device* device() const { return device_.Get(); }
    ID3D12CommandQueue* queue() const { return queue_.Get(); }
    ID3D12CommandAllocator* allocator() const { return allocator_.Get(); }
    ID3D12GraphicsCommandList* cmd_list() const { return cmd_list_.Get(); }
    ID3D12DescriptorHeap* desc_heap() const { return desc_heap_.Get(); }
    ID3D12RootSignature* root_sig() const { return root_sig_.Get(); }
    UINT desc_handle_inc() const { return desc_handle_inc_; }

    // Create a buffer in the default heap with an upload staging resource.
    Dx12Buffer create_buffer(uint64_t size, bool is_uav);

    // Upload data from CPU to a GPU buffer via the upload heap.
    void upload_buffer(const Dx12Buffer& buf, const void* data, uint64_t byte_size);

    // Readback from GPU buffer to CPU (for result verification).
    std::vector<uint8_t> readback_buffer(const Dx12Buffer& buf, uint64_t byte_size);

    // Command list management.
    void reset_command_list();
    void close_command_list();
    void execute_command_list();
    void signal_fence();
    void wait_for_fence();

    // Descriptor heap allocation: returns CPU handle for a free slot.
    D3D12_CPU_DESCRIPTOR_HANDLE alloc_descriptor();

private:
    bool create_root_signature();

    ComPtr<IDXGIFactory6> factory_;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> cmd_list_;
    ComPtr<ID3D12DescriptorHeap> desc_heap_;
    ComPtr<ID3D12RootSignature> root_sig_;
    ComPtr<ID3D12Fence> fence_;
    void* fence_event_ = nullptr;
    UINT64 fence_value_ = 0;
    UINT desc_handle_inc_ = 0;
    UINT desc_heap_count_ = 256;
    UINT desc_heap_index_ = 0;
};

// Compile HLSL source from file to bytecode.
std::vector<uint8_t> compile_hlsl(const std::string& hlsl_path,
                                    const std::string& entry_point,
                                    const std::string& target = "cs_5_0");

// Compile HLSL from in-memory source string.
std::vector<uint8_t> compile_hlsl_source(const std::string& source,
                                         const std::string& entry_point,
                                         const std::string& target = "cs_5_0");

} // namespace dx12

#endif // _WIN32
