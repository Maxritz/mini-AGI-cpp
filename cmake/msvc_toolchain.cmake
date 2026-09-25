# Minimal toolchain file for MSVC + Windows SDK 10.0.28000.0 on this machine.
# Invoke: cmake -B build_dx12 -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/msvc_toolchain.cmake -DENABLE_DX12=ON
set(CMAKE_CXX_COMPILER "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.52.36725/bin/Hostx64/x64/cl.exe")
set(CMAKE_C_COMPILER   "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.52.36725/bin/Hostx64/x64/cl.exe")
set(WIN_KITS "C:/Program Files (x86)/Windows Kits/10")
set(INC_DIRS "${WIN_KITS}/Include/10.0.28000.0/ucrt;${WIN_KITS}/Include/10.0.28000.0/shared;${WIN_KITS}/Include/10.0.28000.0/um")
set(LIB_DIRS "${WIN_KITS}/Lib/10.0.28000.0/ucrt/x64;${WIN_KITS}/Lib/10.0.28000.0/um/x64")
string(REPLACE ";" "\",\"" INC_DIRS_STR "${INC_DIRS}")
string(REPLACE ";" "\",\"" LIB_DIRS_STR "${LIB_DIRS}")
set(CMAKE_CXX_FLAGS "/DWIN32 /D_WINDOWS /D_UNICODE /DUNICODE /std:c++17 /W4 /EHsc /O2 /I\"${INC_DIRS_STR}\"")
set(CMAKE_EXE_LINKER_FLAGS "/LIBPATH:\"${LIB_DIRS_STR}\"")
