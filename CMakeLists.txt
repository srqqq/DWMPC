cmake_minimum_required(VERSION 3.15)   
# set configuration variables
set(PLUGIN_NAME dwmpc)
set(MODULE_LIBRARY_NAME ${PLUGIN_NAME}_module)
set(CONSOLE_COMMANDS_LIBRARY_NAME ${PLUGIN_NAME}_console_commands)
set(MSGS_LIBRARY_NAME ${PLUGIN_NAME}_msgs)
set(MSGS_WRAPPERS_LIBRARY_NAME ${PLUGIN_NAME}_msgs_wrappers)
set(TOPICS_LIBRARY_NAME ${PLUGIN_NAME}_topics)

project(${PLUGIN_NAME})

add_compile_options("-std=c++2a")

set(CMAKE_CXX_FLAGS "-O3")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
find_package(pinocchio REQUIRED)
find_package(hpipm-cpp REQUIRED)
find_package(Threads REQUIRED)

# Create the library for the module
add_library(${MODULE_LIBRARY_NAME} SHARED
	src/${PLUGIN_NAME}.cpp
	src/timer.cpp
	src/codmpc_solver.cpp
	src/pinocchio_model.cpp
)

target_link_directories(${MODULE_LIBRARY_NAME}
    PUBLIC
	/opt/openrobots/lib
	/usr/lib/dls2/controllers/${PLUGIN_NAME}
)

target_compile_definitions(${MODULE_LIBRARY_NAME} PUBLIC CURVES_WITH_PINOCCHIO_SUPPORT)

# 公用配置
target_include_directories(${MODULE_LIBRARY_NAME}
	PUBLIC
	include
	/opt/openrobots/include
)

target_link_libraries(${MODULE_LIBRARY_NAME}
	PUBLIC
	yaml-cpp
	Eigen3::Eigen
	pinocchio::pinocchio
	hpipm-cpp::hpipm-cpp
	Threads::Threads
)

pybind11_add_module(pydwmpc src/py_dwmpc.cpp)
target_link_libraries(pydwmpc
	PUBLIC
	${MODULE_LIBRARY_NAME}
)

# ================================== Install =================================
install(
	DIRECTORY   include/
	DESTINATION /usr/include/dls2
	COMPONENT   ${MODULE_LIBRARY_NAME}_dev
)

install(
	DIRECTORY   config
	DESTINATION /usr/include/dls2/controllers/${PLUGIN_NAME}
	COMPONENT   ${MODULE_LIBRARY_NAME}
)

install(
	DIRECTORY   urdf
	DESTINATION /usr/include/dls2/controllers/${PLUGIN_NAME}
	COMPONENT   ${MODULE_LIBRARY_NAME}
)

install(
	TARGETS             ${MODULE_LIBRARY_NAME}
	LIBRARY DESTINATION /usr/lib/dls2/controllers/${PLUGIN_NAME}
	COMPONENT           ${MODULE_LIBRARY_NAME}
)

install(TARGETS pydwmpc DESTINATION /usr/lib/dls2/controllers/${PLUGIN_NAME})
