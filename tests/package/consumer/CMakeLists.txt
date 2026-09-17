cmake_minimum_required(VERSION 3.24)

project(quant-lstm-cpu-consumer LANGUAGES CXX)

find_package(quant-lstm CONFIG REQUIRED)
if(QUANT_LSTM_WITH_CUDA)
    message(FATAL_ERROR "CPU-only consumer 不得加载 CUDA package")
endif()

add_executable(quant_lstm_cpu_consumer main.cc)
target_link_libraries(
    quant_lstm_cpu_consumer
    PRIVATE quant_lstm::quant_lstm
)
