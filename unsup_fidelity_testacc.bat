@REM for %%k in (1 2 3 4 5 10) do (
@REM     python unsup_fidelity_testacc.py --topk=%%k --train_ratio=0.1
@REM )

for %%k in (0.01 0.02 0.03 0.04 0.05 0.1) do (
    python unsup_fidelity_testacc.py --topk=%%k --train_ratio=0.1
)

pause