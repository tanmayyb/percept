# Sample Workloads

This dir contains sample workloads for:
- planner comparison (planner_comp)
- compute performance comparison (compute_perf)

which were manually tuned to minimize the overall cost by tuning:
- agents
- cost function weights

The generated plans in percept/eval/plans use the coefficients from these sample workloads to generate randomized experiment workloads.


Command to run:
```
ros2 launch mp_eval launch_workload.py disable_metrics:=true timed_run:=0.0 workload:=eval/workloads/XXX.yaml
```