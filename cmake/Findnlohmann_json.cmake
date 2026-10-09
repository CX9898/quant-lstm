find_package(nlohmann_json ${nlohmann_json_FIND_VERSION} CONFIG QUIET)
if(nlohmann_json_FOUND)
    return()
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import nlohmann_json; print(nlohmann_json.get_include())"
        OUTPUT_VARIABLE _json_include
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
find_path(nlohmann_json_INCLUDE_DIR nlohmann/json.hpp HINTS "${_json_include}")
if(nlohmann_json_INCLUDE_DIR)
    file(STRINGS "${nlohmann_json_INCLUDE_DIR}/nlohmann/detail/abi_macros.hpp"
        _json_version_lines REGEX "^#define NLOHMANN_JSON_VERSION_(MAJOR|MINOR|PATCH) ")
    foreach(_part MAJOR MINOR PATCH)
        string(REGEX MATCH "NLOHMANN_JSON_VERSION_${_part} +([0-9]+)" _match "${_json_version_lines}")
        set(_json_${_part} "${CMAKE_MATCH_1}")
    endforeach()
    set(nlohmann_json_VERSION "${_json_MAJOR}.${_json_MINOR}.${_json_PATCH}")
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(nlohmann_json
    REQUIRED_VARS nlohmann_json_INCLUDE_DIR VERSION_VAR nlohmann_json_VERSION)
if(nlohmann_json_FOUND AND NOT TARGET nlohmann_json::nlohmann_json)
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
    set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${nlohmann_json_INCLUDE_DIR}")
endif()
