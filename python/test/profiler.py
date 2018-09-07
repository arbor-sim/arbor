import pyarb as arb

ctx=arb.context()
mm=arb.meter_manager()

mm.start(ctx)

for i in range(1,1000000):
    j=i*2

mm.checkpoint('task1: muliply by 2', ctx)

for i in range(1,1000000):
    k=i+2

mm.checkpoint('task2: add 2', ctx)

report = arb.make_meter_report(mm, ctx)
print(report)
