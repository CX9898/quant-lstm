cmake_minimum_required(VERSION 3.24)

project(quant-lstm VERSION 0.1.0 LANGUAGES CXX)

option(QUANT_LSTM_ENABLE_CUDA "构建 CUDA 浮点前向" ON)
option(QUANT_LSTM_BUILD_TESTS "构建测试" ON)
option(QUANT_LSTM_BUILD_EXAMPLES "构建示例" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(quant_lstm_headers INTERFACE)
target_include_directories(
    quant_lstm_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_library(quant_lstm STATIC src/lstm/forward_float_cpu.cc)
target_link_libraries(quant_lstm PUBLIC quant_lstm_headers)
set_target_properties(quant_lstm PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(quant_lstm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

if(QUANT_LSTM_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "QUANT_LSTM_ENABLE_CUDA=ON，但未找到 CUDA 编译器")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)

    target_sources(quant_lstm PRIVATE src/lstm/forward_float_cuda.cu)
    target_compile_definitions(quant_lstm PUBLIC QUANT_LSTM_WITH_CUDA=1)
    target_link_libraries(quant_lstm PUBLIC CUDA::cublas CUDA::cudart)
    set_target_properties(
        quant_lstm
        PROPERTIES
            CUDA_ARCHITECTURES native
            CUDA_SEPARABLE_COMPILATION OFF
    )
endif()

if(QUANT_LSTM_BUILD_TESTS)
    include(CTest)
    enable_testing()

    add_library(quant_lstm_test_support STATIC tests/common/deterministic_rng.cc)
    target_include_directories(quant_lstm_test_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    set_target_properties(quant_lstm_test_support PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(
        quant_lstm_test_support PRIVATE -Wall -Wextra -Wpedantic
    )

    add_executable(test_gate_layout tests/cpp/test_gate_layout.cc)
    target_link_libraries(test_gate_layout PRIVATE quant_lstm_headers)
    add_test(NAME gate_layout COMMAND test_gate_layout)

    add_executable(test_deterministic_rng tests/cpp/test_deterministic_rng.cc)
    target_link_libraries(test_deterministic_rng PRIVATE quant_lstm_test_support)
    add_test(NAME deterministic_rng COMMAND test_deterministic_rng)

    add_executable(test_float_cpu tests/cpp/test_float_cpu.cc)
    target_link_libraries(test_float_cpu PRIVATE quant_lstm)
    add_test(NAME float_cpu COMMAND test_float_cpu)

    if(QUANT_LSTM_ENABLE_CUDA)
        add_executable(test_float_cuda tests/cuda/test_float_cuda.cu)
        target_link_libraries(test_float_cuda PRIVATE quant_lstm quant_lstm_test_support)
        set_target_properties(test_float_cuda PROPERTIES CUDA_ARCHITECTURES native)
        add_test(NAME float_cuda COMMAND test_float_cuda)
    endif()
endif()

if(QUANT_LSTM_BUILD_EXAMPLES)
    add_executable(lstm_float_example example/lstm_float_example.cc)
    target_link_libraries(lstm_float_example PRIVATE quant_lstm)
endif()
