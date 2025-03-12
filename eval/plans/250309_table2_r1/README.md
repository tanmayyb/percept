# Table 2-R1

### Commands
```
./eval/cleanup.sh eval/plans/250309_table2_r1
python eval/plans/250309_table2_r1/plan.py
python eval/run_plan.py eval/plans/250309_table2_r1/ --duration 0.0
```

### Clear and generate plan
```
clear && ./eval/cleanup.sh eval/plans/250309_table2_r1 && python eval/plans/250309_table2_r1/plan.py
```

### Run plan
```
python eval/run_plan.py eval/plans/250309_table2_r1/ --duration 0.0 --enable_metrics
```
