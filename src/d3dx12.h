// src/d3dx12.h
// Minimal d3dx12.h helper header providing the CD3DX12_* utilities
// used by dx12_context.cpp. This is a minimal implementation sufficient
// for the mini-AGI DX12 backend.

#pragma once

#ifdef _WIN32

#include <d3d12.h>
#include <dxgi.h>
#include <cstdint>
#include <utility>

// Helper structure for read/write range.
struct CD3DX12_RANGE {
    SIZE_T Begin;
    SIZE_T End;
    CD3DX12_RANGE() : Begin(0), End(0) {}
    CD3DX12_RANGE(SIZE_T begin, SIZE_T end) : Begin(begin), End(end) {}
    operator const D3D12_RANGE* () const {
        return reinterpret_cast<const D3D12_RANGE*>(this);
    }
    operator D3D12_RANGE* () {
        return reinterpret_cast<D3D12_RANGE*>(this);
    }
};

// Helper for resource barriers.
struct CD3DX12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER Native;

    CD3DX12_RESOURCE_BARRIER() {
        Native.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        Native.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    }

    static CD3DX12_RESOURCE_BARRIER UAV(ID3D12Resource* pResource,
                                         D3D12_RESOURCE_STATES StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                         D3D12_RESOURCE_STATES StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
        CD3DX12_RESOURCE_BARRIER barrier;
        barrier.Native.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.Native.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Native.UAV.pResource = pResource;
        return barrier;
    }

    static CD3DX12_RESOURCE_BARRIER Transition(ID3D12Resource* pResource,
                                               D3D12_RESOURCE_STATES StateBefore,
                                               D3D12_RESOURCE_STATES StateAfter,
                                               UINT Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                               D3D12_RESOURCE_BARRIER_FLAGS Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE) {
        CD3DX12_RESOURCE_BARRIER barrier;
        barrier.Native.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Native.Flags = Flags;
        barrier.Native.Transition.pResource = pResource;
        barrier.Native.Transition.StateBefore = StateBefore;
        barrier.Native.Transition.StateAfter = StateAfter;
        barrier.Native.Transition.Subresource = Subresource;
        return barrier;
    }

    operator D3D12_RESOURCE_BARRIER* () { return &Native; }
    operator const D3D12_RESOURCE_BARRIER* () const { return &Native; }
};

// Helper for descriptor ranges.
struct CD3DX12_DESCRIPTOR_RANGE {
    D3D12_DESCRIPTOR_RANGE Native;

    CD3DX12_DESCRIPTOR_RANGE() {
        Native.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        Native.NumDescriptors = 0;
        Native.BaseShaderRegister = 0;
        Native.RegisterSpace = 0;
        Native.OffsetInDescriptorsFromTableStart = 0;
    }

    CD3DX12_DESCRIPTOR_RANGE(D3D12_DESCRIPTOR_RANGE_TYPE Type, UINT NumDescriptorsIn,
                             UINT BaseRegister, UINT Space = 0,
                             UINT OffsetFromDescriptorRange = 0) {
        Init(Type, NumDescriptorsIn, BaseRegister, Space, OffsetFromDescriptorRange);
    }

    void Init(D3D12_DESCRIPTOR_RANGE_TYPE Type, UINT NumDescriptorsIn,
              UINT BaseRegister, UINT Space = 0,
              UINT OffsetFromDescriptorRange = 0) {
        Native.RangeType = Type;
        Native.NumDescriptors = NumDescriptorsIn;
        Native.BaseShaderRegister = BaseRegister;
        Native.RegisterSpace = Space;
        Native.OffsetInDescriptorsFromTableStart = OffsetFromDescriptorRange;
    }

    operator const D3D12_DESCRIPTOR_RANGE* () const { return &Native; }
};

// Helper for root parameters.
// Note: This SDK uses D3D12_ROOT_PARAMETER_TYPE_* enums (not SHADER_PARAMETER_TYPE).
struct CD3DX12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER Native;

    CD3DX12_ROOT_PARAMETER() {
        Native.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        Native.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        ZeroMemory(&Native.DescriptorTable, sizeof(Native.DescriptorTable));
    }

    void InitAsDescriptorTable(UINT NumDescriptorRanges,
                               _In_reads_(NumDescriptorRanges) const D3D12_DESCRIPTOR_RANGE* pDescriptorRanges,
                               D3D12_SHADER_VISIBILITY ShaderVisibility) {
        Native.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        Native.ShaderVisibility = ShaderVisibility;
        Native.DescriptorTable.NumDescriptorRanges = NumDescriptorRanges;
        Native.DescriptorTable.pDescriptorRanges = pDescriptorRanges;
    }

    void InitAsConstants(UINT Num32BitValues, UINT ShaderRegister,
                         UINT RegisterSpace = 0,
                         D3D12_SHADER_VISIBILITY ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL) {
        Native.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        Native.ShaderVisibility = ShaderVisibility;
        Native.Constants.Num32BitValues = Num32BitValues;
        Native.Constants.ShaderRegister = ShaderRegister;
        Native.Constants.RegisterSpace = RegisterSpace;
    }

    void InitAsConstantBufferView(UINT ShaderRegister,
                                  UINT RegisterSpace = 0,
                                  D3D12_SHADER_VISIBILITY ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL) {
        Native.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        Native.ShaderVisibility = ShaderVisibility;
        Native.Descriptor.ShaderRegister = ShaderRegister;
        Native.Descriptor.RegisterSpace = RegisterSpace;
    }

    void InitAsShaderResourceView(UINT ShaderRegister,
                                  UINT RegisterSpace = 0,
                                  D3D12_SHADER_VISIBILITY ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL) {
        Native.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        Native.ShaderVisibility = ShaderVisibility;
        Native.Descriptor.ShaderRegister = ShaderRegister;
        Native.Descriptor.RegisterSpace = RegisterSpace;
    }

    void InitAsUnorderedAccessView(UINT ShaderRegister,
                                   UINT RegisterSpace = 0,
                                   D3D12_SHADER_VISIBILITY ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL) {
        Native.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        Native.ShaderVisibility = ShaderVisibility;
        Native.Descriptor.ShaderRegister = ShaderRegister;
        Native.Descriptor.RegisterSpace = RegisterSpace;
    }

    operator const D3D12_ROOT_PARAMETER* () const { return &Native; }
    operator D3D12_ROOT_PARAMETER* () { return &Native; }
};

// Helper for root signature descriptions.
// This SDK uses NumParameters/pParameters (not NumRootParameters/pRootParameters).
struct CD3DX12_ROOT_SIGNATURE_DESC {
    D3D12_ROOT_SIGNATURE_DESC Native;

    CD3DX12_ROOT_SIGNATURE_DESC() {
        Native.NumParameters = 0;
        Native.pParameters = nullptr;
        Native.NumStaticSamplers = 0;
        Native.pStaticSamplers = nullptr;
        Native.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    }

    CD3DX12_ROOT_SIGNATURE_DESC(UINT numRootParameters,
                                _In_reads_opt_(numRootParameters) const D3D12_ROOT_PARAMETER* pRootParameters,
                                UINT numStaticSamplers = 0,
                                _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* pStaticSamplers = nullptr,
                                D3D12_ROOT_SIGNATURE_FLAGS Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) {
        Init(numRootParameters, pRootParameters, numStaticSamplers, pStaticSamplers, Flags);
    }

    void Init(UINT numRootParameters,
              _In_reads_opt_(numRootParameters) const D3D12_ROOT_PARAMETER* pRootParameters,
              UINT numStaticSamplers = 0,
              _In_reads_opt_(numStaticSamplers) const D3D12_STATIC_SAMPLER_DESC* pStaticSamplers = nullptr,
              D3D12_ROOT_SIGNATURE_FLAGS Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE) {
        Native.NumParameters = numRootParameters;
        Native.pParameters = pRootParameters;
        Native.NumStaticSamplers = numStaticSamplers;
        Native.pStaticSamplers = pStaticSamplers;
        Native.Flags = Flags;
    }

    operator const D3D12_ROOT_SIGNATURE_DESC* () const { return &Native; }
    operator D3D12_ROOT_SIGNATURE_DESC* () { return &Native; }
};

#endif // _WIN32
