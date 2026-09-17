cmake_minimum_required(VERSION 3.24)

project(quant-lstm VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(QUANT_LSTM_ENABLE_CUDA "构建 CUDA 浮点前向" ON)
option(QUANT_LSTM_BUILD_TESTS "构建测试" ON)
option(QUANT_LSTM_BUILD_EXAMPLES "构建示例" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(nlohmann_json 3.11.2 REQUIRED)

add_library(quant_lstm_headers INTERFACE)
target_include_directories(
    quant_lstm_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

add_library(
    quant_lstm STATIC
    src/lstm/calibration.cc
    src/lstm/forward_float_cpu.cc
    src/lstm/forward_int32_cpu.cc
    src/lstm/forward_quantized_fp_cpu.cc
    src/lstm/lstm_execution_params.cc
    src/lstm/quant_config.cc
    src/lstm/quant_config_loader.cc
    src/lstm/quant_params.cc
    src/lstm/quant_params_io.cc
    src/quantization/histogram.cc
    src/quantization/numeric_safety.cc
    src/quantization/quant_param.cc
    src/quantization/real_activation.cc
    src/quantization/scale_encoding.cc
)
target_link_libraries(quant_lstm PUBLIC quant_lstm_headers)
target_link_libraries(quant_lstm PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(quant_lstm PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(quant_lstm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>)
add_library(quant_lstm::quant_lstm ALIAS quant_lstm)
add_library(quant_lstm::headers ALIAS quant_lstm_headers)

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

    target_sources(
        quant_lstm
        PRIVATE
            src/lstm/forward_float_cuda.cu
            src/lstm/forward_quantized_fp_cuda.cu
    )
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
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    add_library(
        quant_lstm_test_support
        STATIC
            tests/common/deterministic_rng.cc
            tests/common/synthetic_precision_fixture.cc
    )
    target_include_directories(quant_lstm_test_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(
        quant_lstm_test_support
        PUBLIC quant_lstm nlohmann_json::nlohmann_json
    )
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

    add_executable(test_quantization_primitives tests/cpp/test_quantization_primitives.cc)
    target_link_libraries(test_quantization_primitives PRIVATE quant_lstm)
    add_test(NAME quantization_primitives COMMAND test_quantization_primitives)

    add_executable(test_quant_config tests/cpp/test_quant_config.cc)
    target_link_libraries(test_quant_config PRIVATE quant_lstm)
    target_compile_definitions(
        test_quant_config PRIVATE QUANT_LSTM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME quant_config COMMAND test_quant_config)

    add_executable(test_quant_params tests/cpp/test_quant_params.cc)
    target_link_libraries(test_quant_params PRIVATE quant_lstm)
    target_compile_definitions(
        test_quant_params PRIVATE QUANT_LSTM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME quant_params COMMAND test_quant_params)

    add_executable(test_calibration_minmax tests/cpp/test_calibration_minmax.cc)
    target_link_libraries(test_calibration_minmax PRIVATE quant_lstm)
    target_compile_definitions(
        test_calibration_minmax
        PRIVATE QUANT_LSTM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME calibration_minmax COMMAND test_calibration_minmax)

    add_executable(
        test_calibration_histogram tests/cpp/test_calibration_histogram.cc)
    target_link_libraries(test_calibration_histogram PRIVATE quant_lstm)
    target_compile_definitions(
        test_calibration_histogram
        PRIVATE QUANT_LSTM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME calibration_histogram COMMAND test_calibration_histogram)

    add_executable(test_quant_params_io tests/cpp/test_quant_params_io.cc)
    target_link_libraries(
        test_quant_params_io PRIVATE quant_lstm nlohmann_json::nlohmann_json)
    target_compile_definitions(
        test_quant_params_io
        PRIVATE QUANT_LSTM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME quant_params_io COMMAND test_quant_params_io)

    add_executable(
        test_quantized_references tests/cpp/test_quantized_references.cc)
    target_link_libraries(
        test_quantized_references PRIVATE quant_lstm quant_lstm_test_support)
    add_test(NAME quantized_references COMMAND test_quantized_references)

    add_executable(test_quantized_cell tests/cpp/test_quantized_cell.cc)
    target_link_libraries(test_quantized_cell PRIVATE quant_lstm)
    add_test(NAME quantized_cell COMMAND test_quantized_cell)

    add_executable(test_q31_verification tests/cpp/test_q31_verification.cc)
    target_link_libraries(
        test_q31_verification PRIVATE quant_lstm quant_lstm_test_support)
    add_test(NAME q31_verification COMMAND test_q31_verification)

    add_executable(
        test_synthetic_precision tests/cpp/test_synthetic_precision.cc)
    target_link_libraries(
        test_synthetic_precision
        PRIVATE quant_lstm quant_lstm_test_support nlohmann_json::nlohmann_json)
    target_compile_definitions(
        test_synthetic_precision
        PRIVATE QUANT_LSTM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(
        NAME synthetic_precision
        COMMAND
            test_synthetic_precision
            ${CMAKE_CURRENT_BINARY_DIR}/synthetic_numeric_report.json)

    add_test(
        NAME quant_config_schema
        COMMAND ${Python3_EXECUTABLE} -m unittest tests/python/test_quant_config_schema.py
    )
    set_tests_properties(
        quant_config_schema PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(
        NAME quant_params_bundle_schema
        COMMAND
            ${Python3_EXECUTABLE}
            -m unittest tests/python/test_quant_params_bundle_schema.py
    )
    set_tests_properties(
        quant_params_bundle_schema
        PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(
        NAME pytorch_interface_schemas
        COMMAND
            ${Python3_EXECUTABLE}
            -m unittest tests/python/test_pytorch_interface_schemas.py
    )
    set_tests_properties(
        pytorch_interface_schemas
        PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    add_test(
        NAME precision_contract
        COMMAND ${Python3_EXECUTABLE} -m unittest tests/python/test_precision_contract.py
    )
    set_tests_properties(
        precision_contract PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(
        NAME precision_contract_v2
        COMMAND ${Python3_EXECUTABLE} -m unittest tests/python/test_precision_contract_v2.py
    )
    set_tests_properties(
        precision_contract_v2 PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(
        NAME stage4_cuda_benchmark_contract
        COMMAND
            ${Python3_EXECUTABLE}
            -m unittest tests/python/test_stage4_cuda_benchmark_contract.py
    )
    set_tests_properties(
        stage4_cuda_benchmark_contract
        PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    file(
        GLOB_RECURSE GOLDEN_FILES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/spec/*.json"
    )
    set(GOLDEN_FIXTURE
        "${CMAKE_CURRENT_BINARY_DIR}/generated/golden_fixtures.h")
    add_custom_command(
        OUTPUT ${GOLDEN_FIXTURE}
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_golden.py
            --golden-dir ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/spec
            --schema ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/schema/golden_case.schema.json
            --output ${GOLDEN_FIXTURE}
        DEPENDS
            ${GOLDEN_FILES}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/schema/golden_case.schema.json
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_golden.py
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/strict_jsonschema.py
        VERBATIM
    )
    add_executable(
        test_primitive_golden
        tests/cpp/test_primitive_golden.cc
        ${GOLDEN_FIXTURE}
    )
    target_include_directories(
        test_primitive_golden PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
    target_link_libraries(
        test_primitive_golden PRIVATE quant_lstm nlohmann_json::nlohmann_json)
    add_test(NAME primitive_golden COMMAND test_primitive_golden)

    add_executable(
        test_reference_golden
        tests/cpp/test_reference_golden.cc
        ${GOLDEN_FIXTURE}
    )
    target_include_directories(
        test_reference_golden PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
    target_link_libraries(
        test_reference_golden PRIVATE quant_lstm nlohmann_json::nlohmann_json)
    add_test(NAME reference_golden COMMAND test_reference_golden)

    add_executable(
        golden_reference_generator tools/golden_reference_generator.cc)
    target_link_libraries(
        golden_reference_generator PRIVATE quant_lstm nlohmann_json::nlohmann_json)
    add_test(
        NAME primitive_golden_schema
        COMMAND ${Python3_EXECUTABLE} -m unittest tests/python/test_primitive_golden_schema.py
    )
    set_tests_properties(
        primitive_golden_schema PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(
        NAME golden_fixture_current
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_golden.py
            --golden-dir ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/spec
            --schema ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/schema/golden_case.schema.json
            --output ${GOLDEN_FIXTURE}
            --check
    )

    if(QUANT_LSTM_ENABLE_CUDA)
        add_executable(test_float_cuda tests/cuda/test_float_cuda.cu)
        target_link_libraries(test_float_cuda PRIVATE quant_lstm quant_lstm_test_support)
        set_target_properties(test_float_cuda PROPERTIES CUDA_ARCHITECTURES native)
        add_test(NAME float_cuda COMMAND test_float_cuda)

        add_executable(test_rounding_cuda tests/cuda/test_rounding_cuda.cu)
        target_link_libraries(test_rounding_cuda PRIVATE quant_lstm_headers CUDA::cudart)
        set_target_properties(test_rounding_cuda PROPERTIES CUDA_ARCHITECTURES native)
        add_test(NAME rounding_cuda COMMAND test_rounding_cuda)

        add_executable(
            test_quantized_fp_cuda tests/cuda/test_quantized_fp_cuda.cu
            ${GOLDEN_FIXTURE})
        target_include_directories(
            test_quantized_fp_cuda
            PRIVATE
                ${CMAKE_CURRENT_BINARY_DIR}/generated
                ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        target_link_libraries(
            test_quantized_fp_cuda
            PRIVATE
                quant_lstm
                nlohmann_json::nlohmann_json)
        set_target_properties(
            test_quantized_fp_cuda PROPERTIES CUDA_ARCHITECTURES native)
        add_test(NAME quantized_fp_cuda COMMAND test_quantized_fp_cuda)

        add_executable(
            test_calibration_roundtrip_cuda
            tests/cuda/test_calibration_roundtrip_cuda.cu)
        target_include_directories(
            test_calibration_roundtrip_cuda
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        target_link_libraries(
            test_calibration_roundtrip_cuda PRIVATE quant_lstm)
        target_compile_definitions(
            test_calibration_roundtrip_cuda
            PRIVATE QUANT_LSTM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        set_target_properties(
            test_calibration_roundtrip_cuda
            PROPERTIES CUDA_ARCHITECTURES native)
        add_test(
            NAME calibration_roundtrip_cuda
            COMMAND test_calibration_roundtrip_cuda)

        add_executable(
            test_synthetic_precision_cuda
            tests/cuda/test_synthetic_precision_cuda.cu)
        target_link_libraries(
            test_synthetic_precision_cuda
            PRIVATE
                quant_lstm
                quant_lstm_test_support
                nlohmann_json::nlohmann_json)
        target_compile_definitions(
            test_synthetic_precision_cuda
            PRIVATE QUANT_LSTM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        set_target_properties(
            test_synthetic_precision_cuda PROPERTIES CUDA_ARCHITECTURES native)
        add_test(
            NAME synthetic_precision_cuda_basic
            COMMAND
                test_synthetic_precision_cuda
                --selection basic
                --report
                ${CMAKE_CURRENT_BINARY_DIR}/cuda_synthetic_basic_report.json)
        add_test(
            NAME synthetic_precision_cuda_strict
            COMMAND
                test_synthetic_precision_cuda
                --selection strict
                --report
                ${CMAKE_CURRENT_BINARY_DIR}/cuda_synthetic_strict_report.json)

        add_executable(
            quant_lstm_cuda_benchmark
            tests/benchmarks/quantized_fp_cuda_benchmark.cu)
        target_link_libraries(
            quant_lstm_cuda_benchmark
            PRIVATE
                quant_lstm
                quant_lstm_test_support
                nlohmann_json::nlohmann_json
                CUDA::nvtx3)
        target_compile_definitions(
            quant_lstm_cuda_benchmark
            PRIVATE QUANT_LSTM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        set_target_properties(
            quant_lstm_cuda_benchmark
            PROPERTIES
                CUDA_ARCHITECTURES native
                OUTPUT_NAME quant_lstm_cuda_benchmark
                RUNTIME_OUTPUT_DIRECTORY
                    "${CMAKE_BINARY_DIR}/tests/benchmarks")

    endif()
endif()

if(QUANT_LSTM_BUILD_EXAMPLES)
    add_executable(lstm_float_example example/lstm_float_example.cc)
    target_link_libraries(lstm_float_example PRIVATE quant_lstm)
    add_executable(lstm_int32_example example/lstm_int32_example.cc)
    target_link_libraries(lstm_int32_example PRIVATE quant_lstm)
endif()


install(
    TARGETS quant_lstm quant_lstm_headers
    EXPORT quant_lstmTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
if(QUANT_LSTM_BUILD_EXAMPLES)
    install(
        TARGETS lstm_float_example lstm_int32_example
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)
install(
    FILES
        config/defaults/lstm_quant_default_v1.json
        config/schema/lstm_quant_override.schema.json
        config/schema/lstm_quant_resolved.schema.json
        config/schema/lstm_quant_params_bundle.schema.json
        config/schema/lstm_pytorch_quant_params.schema.json
        config/schema/lstm_pytorch_bidirectional_quant_params.schema.json
    DESTINATION ${CMAKE_INSTALL_DATADIR}/quant-lstm/config
)
install(
    FILES
        README.md
        docs/configuration.md
        docs/dual-carrier-execution.md
        docs/quantized-execution-spec.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/quant-lstm
)


configure_package_config_file(
    cmake/quant-lstm-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/quant-lstm-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quant-lstm
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/quant-lstm-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    EXPORT quant_lstmTargets
    FILE quant-lstm-targets.cmake
    NAMESPACE quant_lstm::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quant-lstm
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/quant-lstm-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/quant-lstm-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quant-lstm
)
