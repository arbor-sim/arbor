# Convenience target that builds all examples.
# Example executable targets should be added to the 'examples' target as dependencies.
add_custom_target(examples DEPENDS)

add_subdirectory(drybench)
add_subdirectory(dryrun)
add_subdirectory(generators)
add_subdirectory(brunel)
add_subdirectory(bench)
add_subdirectory(ring)
add_subdirectory(gap_junctions)
add_subdirectory(single)
add_subdirectory(probe-demo)
add_subdirectory(plasticity)
add_subdirectory(lfp)
add_subdirectory(diffusion)
add_subdirectory(v_clamp)
add_subdirectory(ornstein_uhlenbeck)
