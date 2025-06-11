# 250510 - Compute Performance Table R1

### Commands
```
./eval/cleanup.sh eval/plans/compute_perf_r1
python eval/plans/compute_perf_r1/plan.py
python eval/run_plan.py eval/plans/compute_perf_r1 --duration 0.0
```

### Clear and generate plan
```
clear && ./eval/cleanup.sh eval/plans/compute_perf_r1 && python eval/plans/compute_perf_r1/plan.py
```

### Run plan
```
python eval/run_plan.py eval/plans/compute_perf_r1/plan.py --duration 0.0 --enable_metrics
```
