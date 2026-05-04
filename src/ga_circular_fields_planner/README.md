## GA Circular Fields Planner


### Adding New Heuristic Forces

To add a new heuristic force, you need to:

1. Create a new cpp and header file in the `src/ga_circular_fields_planner` like other heuristic forces.
2. Change `REGISTER_CLASS` label in the new cpp file to the new heuristic force name.
3. Change the topic name in the cpp file to the new heuristic force topic name.
4. Add cpp and header to the `src/ga_circular_fields_planner/CMakeLists.txt` file.


### Adding New Cost Functions

To add a new cost function, you need to:

1. Create the HPP and the CPP in the /src folder and define interface specification (if ROS service requests needed).
2. Add cpp and header to the `src/ga_circular_fields_planner/CMakeLists.txt` file.
3. Modify the experiments config yaml accordingly to enable the new cost function in the planner.