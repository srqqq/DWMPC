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

# ！！！下面的求解器只能开启一个，请勿同时开启多个！！！
option(USE_HPIPM   "Enable HPIPM solver"   ON)
option(USE_FPGA    "Enable FPGA solver"    OFF)
option(USE_QPOASES "Enable qpOASES solver" OFF) # qpOASES已废弃，请勿使用！！！

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
find_package(pinocchio REQUIRED)
# add_subdirectory(third_party/acados)

# Create the library for the module
add_library(${MODULE_LIBRARY_NAME} SHARED
	src/${PLUGIN_NAME}.cpp
	# src/acadosInterface.cpp
	src/timer.cpp
	# src/distributed_solver.cpp
	src/codmpc_solver.cpp
	src/pinocchio_model.cpp
	src/robot_data_logger.cpp
	# add other source files here
)

target_link_directories(${MODULE_LIBRARY_NAME}
    PUBLIC
	# ${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code
	/opt/openrobots/lib
	/usr/lib/dls2/controllers/${PLUGIN_NAME}
)

# 添加预处理器定义
target_compile_definitions(${MODULE_LIBRARY_NAME} PUBLIC CURVES_WITH_PINOCCHIO_SUPPORT)
target_compile_definitions(${MODULE_LIBRARY_NAME} PUBLIC DEBUG_MODE)

# 公用配置
target_include_directories(${MODULE_LIBRARY_NAME}
	PUBLIC
	include
	# ${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code
	/opt/openrobots/include
	# add other include directories here
)

target_link_libraries(${MODULE_LIBRARY_NAME}
	PUBLIC
	# acados
	# ${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code/libacados_ocp_solver_front.so
	# ${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code/libacados_ocp_solver_back.so
	yaml-cpp
	Eigen3::Eigen
	pinocchio::pinocchio
	# add other linked libraries here
)

########## USE_HPIPM ##########
if (USE_HPIPM)
	target_compile_definitions(${MODULE_LIBRARY_NAME} PUBLIC USE_HPIPM) # 定义宏
	find_package(hpipm-cpp REQUIRED)

	target_include_directories(${MODULE_LIBRARY_NAME}
		PUBLIC
		${hpipm-cpp_INCLUDE_DIR}
	)

	target_link_libraries(${MODULE_LIBRARY_NAME}
		PUBLIC
		hpipm-cpp::hpipm-cpp
	)
endif()
########################################

########## USE_FPGA ##########
if (USE_FPGA)
	target_compile_definitions(${MODULE_LIBRARY_NAME} PUBLIC USE_FPGA) # 定义宏
	find_package(Threads REQUIRED)

	target_link_libraries(${MODULE_LIBRARY_NAME}
		PUBLIC
		fish_protocol  # 链接libfish_protocol库
		Threads::Threads  # 链接pthread库（等效于-lpthread）
	)
endif()
########################################

########## USE_QPOASES ##########
if (USE_QPOASES)
	target_compile_definitions(${MODULE_LIBRARY_NAME} PUBLIC USE_QPOASES) # 定义宏
	find_package(qpOASES REQUIRED)

	target_link_libraries(${MODULE_LIBRARY_NAME}
		PUBLIC
		qpOASES
	)
endif()
########################################

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

# install(
# 	FILES       ${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code/libacados_ocp_solver_front.so
# 				${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code/libacados_ocp_solver_back.so
# 	COMPONENT   ${MODULE_LIBRARY_NAME}
# 	DESTINATION /usr/lib/dls2/controllers/${PLUGIN_NAME}
# )
install(TARGETS pydwmpc DESTINATION /usr/lib/dls2/controllers/${PLUGIN_NAME})