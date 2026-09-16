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
endif()

if(QUANT_LSTM_BUILD_TESTS)
    include(CTest)
    enable_testing()

    add_executable(test_gate_layout tests/cpp/test_gate_layout.cc)
    target_link_libraries(test_gate_layout PRIVATE quant_lstm_headers)
    add_test(NAME gate_layout COMMAND test_gate_layout)
endif()

if(QUANT_LSTM_BUILD_EXAMPLES)
    add_executable(lstm_layout_example example/lstm_layout_example.cc)
    target_link_libraries(lstm_layout_example PRIVATE quant_lstm_headers)
endif()
