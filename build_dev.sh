clear 

# echo "Sourcing underlay from: $percept_ul"

source $percept_ul 

export LDFLAGS='-Wl,--no-as-needed'

colcon build \
	--packages-select \
		percept_core \
		experiments \
		ga_circular_fields_planner \
	--symlink-install \
	--cmake-args \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DCMAKE_CUDA_COMPILER=nvcc \
		-DCMAKE_CUDA_HOST_COMPILER=clang++ \
		-DCMAKE_BUILD_TYPE=Debug \
		-DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer -w" \
	--packages-ignore \
		mp_eval \
		percept_interfaces \
		sackmesser \
		sackmesser_ros2 \
		gafro \
		gafro_ros2 \
		gafro_robot_descriptions




# export LDFLAGS='-Wl,--no-as-needed'

# colcon build \
# 	--symlink-install \
# 	--cmake-args \
# 		-DCMAKE_CXX_COMPILER=clang++ \
# 		-DCMAKE_CUDA_COMPILER=nvcc \
# 		-DCMAKE_CUDA_HOST_COMPILER=clang++ \
# 		-DCMAKE_BUILD_TYPE=Debug \
# 		-DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer -w"



