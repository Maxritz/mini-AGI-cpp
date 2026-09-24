cmake_minimum_required(VERSION 3.15)
project(minimagi LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

# Enable DX12 on Windows (MSVC); off on other platforms.
if(WIN32 AND MSVC)
    option(ENABLE_DX12 "Enable DirectX 12 backend (Windows/MSVC only)" ON)
else()
    set(ENABLE_DX12 OFF CACHE BOOL "Enable DirectX 12 backend" FORCE)
endif()

if(MSVC)
    add_compile_options(/W4 /EHsc /O2)
else()
    add_compile_options(-Wall -Wextra -O2)
endif()
enable_testing()

file(GLOB CORE_SRCS CONFIGURE_DEPENDS src/*.cpp)

# Exclude dx12_dense.cpp from the base library — it's only built when DX12 is enabled.
list(REMOVE_ITEM CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/dx12_dense.cpp)
list(REMOVE_ITEM CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/dx12_pool.cpp)

# Exclude dx12 context/engine from base — these are Windows-only.
list(REMOVE_ITEM CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/dx12_context.cpp)
list(REMOVE_ITEM CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/dx12_engine.cpp)

add_library(minimagi STATIC ${CORE_SRCS})
target_include_directories(minimagi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- DX12 library (Windows MSVC only) ----
if(ENABLE_DX12)
    add_library(dx12_backend STATIC
        src/dx12_context.cpp
        src/dx12_engine.cpp
        src/dx12_dense.cpp
        src/dx12_pool.cpp
    )
    target_include_directories(dx12_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(dx12_backend PUBLIC minimagi)
    target_link_libraries(dx12_backend PRIVATE d3d12 dxgi d3dcompiler)

    # d3dx12.h helper header — search common VS include dirs
    # (The headers use CD3DX12_* helpers which require d3dx12.h. If the system
    #  has the Windows SDK include paths, they should already resolve.)
    target_compile_definitions(dx12_backend PRIVATE NOMINMAX)
endif()

# ---- Test executables ----

add_executable(test_mininpz tests/test_mininpz.cpp)
target_link_libraries(test_mininpz PRIVATE minimagi)
add_test(NAME test_mininpz COMMAND test_mininpz)

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE minimagi)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_config_tokenizer tests/test_config_tokenizer.cpp)
target_link_libraries(test_config_tokenizer PRIVATE minimagi)
add_test(NAME test_config_tokenizer COMMAND test_config_tokenizer)

add_executable(test_store tests/test_store.cpp)
target_link_libraries(test_store PRIVATE minimagi)
add_test(NAME test_store COMMAND test_store)

add_executable(test_model tests/test_model.cpp)
target_link_libraries(test_model PRIVATE minimagi)
add_test(NAME test_model COMMAND test_model)

# ---- Model creation tool ----
add_executable(minagi_create tools/create.cpp)
target_link_libraries(minagi_create PRIVATE minimagi)

if(ENABLE_DX12)
    target_sources(minagi_create PRIVATE src/dx12_context.cpp src/dx12_engine.cpp)
    target_link_libraries(minagi_create PRIVATE d3d12 dxgi d3dcompiler)
    target_compile_definitions(minagi_create PRIVATE NOMINMAX)
endif()

# ---- Decode (inference) tool ----
add_executable(minagi_decode tools/decode.cpp)
target_link_libraries(minagi_decode PRIVATE minimagi)

if(ENABLE_DX12)
    target_sources(minagi_decode PRIVATE src/dx12_context.cpp src/dx12_engine.cpp)
    target_link_libraries(minagi_decode PRIVATE d3d12 dxgi d3dcompiler)
    target_compile_definitions(minagi_decode PRIVATE NOMINMAX)
endif()

add_executable(test_wavef tests/test_wavef.cpp)
target_link_libraries(test_wavef PRIVATE minimagi)
add_test(NAME test_wavef COMMAND test_wavef ..)
set_tests_properties(test_wavef PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)

# ---- DX12 smoke test (Windows MSVC only) ----
if(ENABLE_DX12)
    add_executable(dx12_smoke tests/test_dx12_smoke.cpp)
    target_link_libraries(dx12_smoke PRIVATE minimagi dx12_backend d3d12 dxgi d3dcompiler)
    add_test(NAME dx12_smoke COMMAND dx12_smoke ..)
    set_tests_properties(dx12_smoke PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()
