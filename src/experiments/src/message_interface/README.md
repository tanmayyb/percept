## Message Interface

This is a simple interface for sending and receiving messages between the GA CF Planner and the Heuristic Force Service Servers.

### Adding New Heuristic Forces

1. Create a new cpp and header file in the `src/message_interface` folder.
2. Change `REGISTER_CLASS` label in the new cpp file to the new heuristic force name.
3. Add cpp and header to the `src/message_interface/CMakeLists.txt` file.