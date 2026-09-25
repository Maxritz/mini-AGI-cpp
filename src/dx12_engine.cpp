// src/dx12_engine.cpp
// Implementation of Dx12ComputeEngine. Guarded by #ifdef _WIN32.

#ifdef _WIN32

#include "dx12_engine.hpp"
#include <filesystem>
#include <stdexcept>
#include <iostream>

namespace dx12 {

Dx12ComputeEngine::Dx12ComputeEngine() = default;
Dx12ComputeEngine::~Dx12ComputeEngine() = default;

bool Dx12ComputeEngine::init(const std::string& shader_dir) {
    if (!ctx_.init()) return false;
    shader_dir_ = shader_dir;
    ready_ = true;
    return true;
}

Dx12Buffer Dx12ComputeEngine::create_buffer(uint64_t byte_size, bool is_uav) {
    return ctx_.create_buffer(byte_size, is_uav);
}

void Dx12ComputeEngine::upload_buffer(const Dx12Buffer& buf, const mt::Tensor& tensor) {
    const uint64_t byte_size = static_cast<uint64_t>(tensor.data_.size());
    ctx_.upload_buffer(buf, tensor.data_.data(), byte_size);
}

mt::Tensor Dx12ComputeEngine::readback_buffer(const Dx12Buffer& buf, mt::Shape shape, mt::DType dtype) {
    const uint64_t byte_size = static_cast<uint64_t>(shape.numel()) *
        (dtype == mt::DType::FP32 ? 4 : 2);
    std::vector<uint8_t> bytes = ctx_.readback_buffer(buf, byte_size);
    mt::Tensor t;
    t.dtype = dtype;
    t.shape = shape;
    t.data_ = std::move(bytes);
    return t;
}

void Dx12ComputeEngine::reset() {
    ctx_.reset_command_list();
}

void Dx12ComputeEngine::close() {
    ctx_.close_command_list();
}

void Dx12ComputeEngine::dispatch(uint32_t x, uint32_t y, uint32_t z) {
    cmd_list()->Dispatch(x, y, z);
}

void Dx12ComputeEngine::flush() {
    close();
    ID3D12CommandList* lists[] = { cmd_list() };
    queue()->ExecuteCommandLists(1, lists);
    signal_fence();
    wait_for_fence();
}

void Dx12ComputeEngine::signal_fence() {
    ctx_.signal_fence();
}

void Dx12ComputeEngine::wait_for_fence() {
    ctx_.wait_for_fence();
}

void Dx12ComputeEngine::set_uav(UINT slot, const Dx12Buffer& buf) {
    cmd_list()->SetComputeRootUnorderedAccessView(slot, buf.gpu_address);
}

void Dx12ComputeEngine::set_srv(UINT slot, const Dx12Buffer& buf) {
    // With UAV-only root signature, SRV buffers are also bound as UAVs (reads allowed in compute)
    cmd_list()->SetComputeRootUnorderedAccessView(slot, buf.gpu_address);
}

void Dx12ComputeEngine::push_constants(const void* data, uint32_t num_32bit_values) {
    cmd_list()->SetComputeRoot32BitConstants(12, num_32bit_values,
        static_cast<const uint32_t*>(data), 0);
}

int Dx12ComputeEngine::load_compute_pso(const std::string& entry_point) {
    std::string hlsl_path = shader_dir_ + "/dense.hlsl";
    if (entry_point.find("pool") != std::string::npos) {
        hlsl_path = shader_dir_ + "/pool.hlsl";
    } else if (entry_point.find("grad_") != std::string::npos) {
        hlsl_path = shader_dir_ + "/backward.hlsl";
    }

    std::vector<uint8_t> bytecode;
    try {
        bytecode = compile_hlsl(hlsl_path, entry_point);
    } catch (const std::exception& e) {
        std::cerr << "DX12_ENGINE: PSO compile failed for '" << entry_point
                  << "' from '" << hlsl_path << "': " << e.what() << std::endl;
        return -1;
    }

    if (bytecode.empty()) {
        std::cerr << "DX12_ENGINE: PSO bytecode empty for '" << entry_point << "'" << std::endl;
        return -1;
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC psodesc = {};
    psodesc.pRootSignature = root_sig();
    psodesc.CS.pShaderBytecode = bytecode.data();
    psodesc.CS.BytecodeLength = bytecode.size();
    psodesc.NodeMask = 1;
    psodesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

    PsoEntry entry;
    entry.name = entry_point;
    HRESULT hr = device()->CreateComputePipelineState(&psodesc, IID_PPV_ARGS(&entry.pso));
    if (FAILED(hr)) return -1;

    psos_.push_back(std::move(entry));
    return static_cast<int>(psos_.size()) - 1;
}

void Dx12ComputeEngine::set_pso(int pso_index) {
    if (pso_index < 0 || pso_index >= static_cast<int>(psos_.size())) return;
    cmd_list()->SetPipelineState(psos_[pso_index].pso.Get());
    cmd_list()->SetComputeRootSignature(root_sig());
}

void Dx12ComputeEngine::set_pso(const std::string& name) {
    for (size_t i = 0; i < psos_.size(); ++i) {
        if (psos_[i].name == name) {
            set_pso(static_cast<int>(i));
            return;
        }
    }
}

} // namespace dx12

#endif // _WIN32
