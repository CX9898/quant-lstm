#pragma once

#include "lstm/lstm_execution_params.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace quant_lstm::test {

inline std::string_view fp32AccumulationClassName(
    quantization::Fp32AccumulationClass value) {
    switch (value) {
        case quantization::Fp32AccumulationClass::ExactIntegerRange:
            return "exact_integer_range";
        case quantization::Fp32AccumulationClass::PrecisionRisk:
            return "precision_risk";
        case quantization::Fp32AccumulationClass::UnsafeNonFinite:
            return "unsafe_non_finite";
    }
    throw std::invalid_argument("非法 FP32 accumulation class");
}

inline nlohmann::json numericSafetyReportJson(
    const quantization::NumericSafetyReport& report) {
    return {
        {"classification", fp32AccumulationClassName(report.fp32_accumulation)},
        {"safe", report.safe()},
        {"proofs",
         {
             {"int64_gemm_safe", report.int64_gemm_safe},
             {"bias_rescale_safe", report.bias_rescale_safe},
             {"int64_linear_accumulation_safe",
              report.int64_linear_accumulation_safe},
             {"linear_output_rescale_safe",
              report.linear_output_rescale_safe},
             {"int64_linear_boundary_safe",
              report.int64_linear_boundary_safe},
             {"int64_cell_product_safe", report.int64_cell_product_safe},
             {"int128_cell_safe", report.int128_cell_safe},
             {"int64_cell_requantize_safe",
              report.int64_cell_requantize_safe},
             {"int64_cell_boundary_safe", report.int64_cell_boundary_safe},
             {"int64_hidden_product_safe",
              report.int64_hidden_product_safe},
             {"hidden_rescale_safe", report.hidden_rescale_safe},
             {"int64_hidden_boundary_safe",
              report.int64_hidden_boundary_safe},
             {"gate_lhs_rescale_safe", report.gate_lhs_rescale_safe},
             {"gate_rhs_rescale_safe", report.gate_rhs_rescale_safe},
             {"int64_gate_merge_safe", report.int64_gate_merge_safe},
             {"int64_gate_boundary_safe",
              report.int64_gate_boundary_safe},
             {"encodings_valid", report.encodings_valid},
         }},
        {"bounds",
         {
             {"gemm", report.gemm_bound},
             {"bias_rescale", report.bias_rescale_bound},
             {"linear_accumulation", report.linear_accumulation_bound},
             {"linear_output_rescale",
              report.linear_output_rescale_bound},
             {"linear_boundary", report.linear_boundary_bound},
             {"cell_product", report.cell_product_bound},
             {"cell_q31", report.cell_q31_bound},
             {"cell_requantize", report.cell_requantize_bound},
             {"cell_boundary", report.cell_boundary_bound},
             {"hidden_product", report.hidden_product_bound},
             {"hidden_rescale", report.hidden_rescale_bound},
             {"hidden_boundary", report.hidden_boundary_bound},
             {"gate_lhs_rescale", report.gate_lhs_rescale_bound},
             {"gate_rhs_rescale", report.gate_rhs_rescale_bound},
             {"gate_merge", report.gate_merge_bound},
             {"gate_boundary", report.gate_boundary_bound},
         }},
    };
}

inline nlohmann::json executionSafetyJson(
    const LstmExecutionParams& execution) {
    nlohmann::json input_hidden = nlohmann::json::array();
    for (std::size_t channel = 0;
         channel < execution.diagnostics.input_hidden_linear.size();
         ++channel) {
        auto report = numericSafetyReportJson(
            execution.diagnostics.input_hidden_linear[channel]);
        report["channel"] = channel;
        input_hidden.push_back(std::move(report));
    }
    nlohmann::json hidden_hidden = nlohmann::json::array();
    for (std::size_t channel = 0;
         channel < execution.diagnostics.hidden_hidden_linear.size();
         ++channel) {
        auto report = numericSafetyReportJson(
            execution.diagnostics.hidden_hidden_linear[channel]);
        report["channel"] = channel;
        hidden_hidden.push_back(std::move(report));
    }
    nlohmann::json gates = nlohmann::json::array();
    for (std::size_t gate = 0; gate < execution.diagnostics.gates.size();
         ++gate) {
        auto report =
            numericSafetyReportJson(execution.diagnostics.gates[gate]);
        report["gate"] = gate;
        gates.push_back(std::move(report));
    }
    return {
        {"input_hidden_linear", std::move(input_hidden)},
        {"hidden_hidden_linear", std::move(hidden_hidden)},
        {"gates", std::move(gates)},
        {"cell", numericSafetyReportJson(execution.diagnostics.cell)},
        {"hidden", numericSafetyReportJson(execution.diagnostics.hidden)},
    };
}

}  // namespace quant_lstm::test
