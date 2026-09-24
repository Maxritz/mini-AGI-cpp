// src/dx12_context.cpp
// Implementation of the DX12 context. Guarded by #ifdef _WIN32.

#ifdef _WIN32

#include "dx12_context.hpp"
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <windows.h>
#include <cstring>
#include <stdexcept>

// d3dx12.h provides CD3DX12_* helper types. It is shipped with the Windows SDK
// (under Include\um) and the DirectX SDK samples.
// We try to include it; if unavailable, we fall back to manual helper code.
#include "d3dx12.h"

namespace dx12 {

Dx12Context::Dx12Context() = default;

Dx12Context::~Dx12Context() {
    if (fence_event_) {
        CloseHandle(fence_event_);
        fence_event_ = nullptr;
    }
}

bool Dx12Context::init() {
    // Create DXGI factory
    UINT factory_flags = 0;
#ifdef _DEBUG
    factory_flags = DXGI_CREATE_FACTORY_DEBUG;
#endif
    HRESULT hr = CreateDXGIFactory2(factory_flags, IID_PPV_ARGS(&factory_));
    if (FAILED(hr)) return false;

    // Create D3D12 device — try hardware first, then WARP fallback.
    hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_));
    if (FAILED(hr)) {
        // Try WARP
        ComPtr<IDXGIAdapter> warp_adapter;
        hr = factory_->EnumWarpAdapter(IID_PPV_ARGS(&warp_adapter));
        if (FAILED(hr)) return false;
        hr = D3D12CreateDevice(warp_adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                               IID_PPV_ARGS(&device_));
        if (FAILED(hr)) return false;
    }

    // Create command queue (compute)
    D3D12_COMMAND_QUEUE_DESC qdesc = {};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    qdesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    hr = device_->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue_));
    if (FAILED(hr)) return false;

    // Create command allocator
    hr = device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                         IID_PPV_ARGS(&allocator_));
    if (FAILED(hr)) return false;

    // Create command list
    hr = device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                    allocator_.Get(), nullptr,
                                    IID_PPV_ARGS(&cmd_list_));
    if (FAILED(hr)) return false;
    // Close immediately — caller resets per-frame.
    cmd_list_->Close();

    // Create fence
    hr = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
    if (FAILED(hr)) return false;
    fence_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event_) return false;

    // Create descriptor heap for CBV_SRV_UAV (256 descriptors, shader-visible)
    D3D12_DESCRIPTOR_HEAP_DESC dhdesc = {};
    dhdesc.NumDescriptors = desc_heap_count_;
    dhdesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    dhdesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr = device_->CreateDescriptorHeap(&dhdesc, IID_PPV_ARGS(&desc_heap_));
    if (FAILED(hr)) return false;
    desc_handle_inc_ = device_->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Create root signature
    if (!create_root_signature()) return false;

    return true;
}

bool Dx12Context::create_root_signature() {
    // Root signature:
    //   Root params 0-11: root descriptors for UAV bindings u0-u11
    //   Root param 12: 32-bit root constants (push constants)
    //
    // All buffers are bound as UAV (u0-u11). Compute shaders can read from
    // UAVs, so RWByteAddressBuffer in HLSL handles both read and write.

    CD3DX12_ROOT_PARAMETER root_params[13];

    for (UINT i = 0; i < 12; ++i) {
        root_params[i].InitAsUnorderedAccessView(i, 0,
            D3D12_SHADER_VISIBILITY_ALL);
    }

    // Root constants (32-bit root constants) at root param 12
    root_params[12].InitAsConstants(20, 0, D3D12_SHADER_VISIBILITY_ALL);

    CD3DX12_ROOT_SIGNATURE_DESC rs_desc;
    rs_desc.Init(13, &root_params[0].Native, 0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_NONE);

    ComPtr<ID3DBlob> serialized_sig;
    ComPtr<ID3DBlob> error_msg;
    HRESULT hr = D3D12SerializeRootSignature(
        &rs_desc.Native, D3D_ROOT_SIGNATURE_VERSION_1,
        &serialized_sig, &error_msg);
    if (FAILED(hr)) return false;

    hr = device_->CreateRootSignature(0,
        serialized_sig->GetBufferPointer(),
        serialized_sig->GetBufferSize(),
        IID_PPV_ARGS(&root_sig_));
    return SUCCEEDED(hr);
}

Dx12Buffer Dx12Context::create_buffer(uint64_t size, bool is_uav) {
    Dx12Buffer buf;
    buf.size = size;
    buf.is_uav = is_uav;

    D3D12_HEAP_PROPERTIES default_props = {};
    default_props.Type = D3D12_HEAP_TYPE_DEFAULT;
    default_props.CreationNodeMask = 1;
    default_props.VisibleNodeMask = 1;

    D3D12_HEAP_PROPERTIES upload_props = {};
    upload_props.Type = D3D12_HEAP_TYPE_UPLOAD;
    upload_props.CreationNodeMask = 1;
    upload_props.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC buf_desc = {};
    buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf_desc.Alignment = 0;
    buf_desc.Width = size;
    buf_desc.Height = 1;
    buf_desc.DepthOrArraySize = 1;
    buf_desc.MipLevels = 1;
    buf_desc.Format = DXGI_FORMAT_UNKNOWN;
    buf_desc.SampleDesc.Count = 1;
    buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buf_desc.Flags = is_uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS :
                              D3D12_RESOURCE_FLAG_NONE;

    // Create default heap resource (GPU-local)
    HRESULT hr = device_->CreateCommittedResource(
        &default_props, D3D12_HEAP_FLAG_NONE,
        &buf_desc,
        is_uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS :
                 D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(&buf.default_heap));
    if (FAILED(hr)) return buf;

    buf.gpu_address = buf.default_heap->GetGPUVirtualAddress();

    // Create upload heap (CPU-writable staging)
    hr = device_->CreateCommittedResource(
        &upload_props, D3D12_HEAP_FLAG_NONE,
        &buf_desc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr, IID_PPV_ARGS(&buf.upload_heap));
    if (FAILED(hr)) {
        buf.upload_heap.Reset();
        return buf;
    }

    // Map upload heap for CPU access
    D3D12_RANGE read_range = {0, 0};
    hr = buf.upload_heap->Map(0, &read_range, &buf.cpu_address);
    if (FAILED(hr)) buf.cpu_address = nullptr;

    return buf;
}

void Dx12Context::upload_buffer(const Dx12Buffer& buf, const void* data, uint64_t byte_size) {
    if (!buf.cpu_address || byte_size > buf.size) return;
    std::memcpy(buf.cpu_address, data, byte_size);

    // Copy upload heap -> default heap
    reset_command_list();
    cmd_list_->CopyBufferRegion(buf.default_heap.Get(), 0,
                                buf.upload_heap.Get(), 0, byte_size);
    // Transition to UAV state if needed
    if (buf.is_uav) {
        CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(buf.default_heap.Get());
        cmd_list_->ResourceBarrier(1, &barrier.Native);
    }
    close_command_list();
    execute_command_list();
    signal_fence();
    wait_for_fence();
}

std::vector<uint8_t> Dx12Context::readback_buffer(const Dx12Buffer& buf, uint64_t byte_size) {
    std::vector<uint8_t> result(byte_size, 0);
    if (!buf.gpu_address || byte_size > buf.size) return result;

    // Create a readback buffer (CPU-readable)
    D3D12_HEAP_PROPERTIES rb_props = {};
    rb_props.Type = D3D12_HEAP_TYPE_READBACK;
    rb_props.CreationNodeMask = 1;
    rb_props.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC rb_desc = {};
    rb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rb_desc.Width = byte_size;
    rb_desc.Height = 1;
    rb_desc.DepthOrArraySize = 1;
    rb_desc.MipLevels = 1;
    rb_desc.Format = DXGI_FORMAT_UNKNOWN;
    rb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> readback;
    HRESULT hr = device_->CreateCommittedResource(
        &rb_props, D3D12_HEAP_FLAG_NONE,
        &rb_desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(&readback));
    if (FAILED(hr)) return result;

    // Copy GPU buffer -> readback
    reset_command_list();
    cmd_list_->CopyBufferRegion(readback.Get(), 0,
                                buf.default_heap.Get(), 0, byte_size);
    close_command_list();
    execute_command_list();
    signal_fence();
    wait_for_fence();

    // Map and read
    void* mapped = nullptr;
    D3D12_RANGE read_range = {0, byte_size};
    hr = readback->Map(0, &read_range, &mapped);
    if (SUCCEEDED(hr) && mapped) {
        std::memcpy(result.data(), mapped, byte_size);
        readback->Unmap(0, nullptr);
    }

    return result;
}

void Dx12Context::reset_command_list() {
    allocator_->Reset();
    cmd_list_->Reset(allocator_.Get(), nullptr);
}

void Dx12Context::close_command_list() {
    cmd_list_->Close();
}

void Dx12Context::execute_command_list() {
    ID3D12CommandList* lists[] = { cmd_list_.Get() };
    queue_->ExecuteCommandLists(1, lists);
}

void Dx12Context::signal_fence() {
    fence_value_++;
    queue_->Signal(fence_.Get(), fence_value_);
}

void Dx12Context::wait_for_fence() {
    if (fence_->GetCompletedValue() < fence_value_) {
        fence_->SetEventOnCompletion(fence_value_, fence_event_);
        WaitForSingleObject(fence_event_, INFINITE);
    }
}

D3D12_CPU_DESCRIPTOR_HANDLE Dx12Context::alloc_descriptor() {
    UINT offset = desc_heap_index_ * desc_handle_inc_;
    D3D12_CPU_DESCRIPTOR_HANDLE handle = desc_heap_->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += offset;
    desc_heap_index_ = (desc_heap_index_ + 1) % desc_heap_count_;
    return handle;
}

std::vector<uint8_t> compile_hlsl(const std::string& hlsl_path,
                                   const std::string& entry_point,
                                   const std::string& target) {
    std::wstring wpath(hlsl_path.begin(), hlsl_path.end());

    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> errors;

    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    flags |= D3DCOMPILE_DEBUG;
#endif

    HRESULT hr = D3DCompileFromFile(
        wpath.c_str(),
        nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entry_point.c_str(),
        target.c_str(),
        flags, 0,
        &bytecode, &errors);

    if (FAILED(hr)) {
        if (errors) {
            std::string msg = reinterpret_cast<char*>(errors->GetBufferPointer());
            throw std::runtime_error("HLSL compile error: " + msg);
        }
        throw std::runtime_error("D3DCompileFromFile failed");
    }

    return std::vector<uint8_t>(
        static_cast<uint8_t*>(bytecode->GetBufferPointer()),
        static_cast<uint8_t*>(bytecode->GetBufferPointer()) + bytecode->GetBufferSize());
}

std::vector<uint8_t> compile_hlsl_source(const std::string& source,
                                          const std::string& entry_point,
                                          const std::string& target) {
    ComPtr<ID3DBlob> bytecode;
    ComPtr<ID3DBlob> errors;

    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    flags |= D3DCOMPILE_DEBUG;
#endif

    HRESULT hr = D3DCompile(
        source.data(), source.size(),
        nullptr, nullptr,
        D3D_COMPILE_STANDARD_FILE_INCLUDE,
        entry_point.c_str(),
        target.c_str(),
        flags, 0,
        &bytecode, &errors);

    if (FAILED(hr)) {
        if (errors) {
            std::string msg = reinterpret_cast<char*>(errors->GetBufferPointer());
            throw std::runtime_error("HLSL compile error: " + msg);
        }
        throw std::runtime_error("D3DCompile failed");
    }

    return std::vector<uint8_t>(
        static_cast<uint8_t*>(bytecode->GetBufferPointer()),
        static_cast<uint8_t*>(bytecode->GetBufferPointer()) + bytecode->GetBufferSize());
}

} // namespace dx12

#endif // _WIN32
