#import cynestmc as nestmc
#import pynestmc as nestmc

recipe = nestmc.Recipe(...)
model = nestmc.Model(recipe, ...)
time = model.run(dt, tf)
print("Num spikes: ", model.num_spikes)
