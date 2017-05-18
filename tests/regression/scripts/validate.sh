#####################################################
# Run nest with the standard settings, but output to file
echo "running rgraph regression test"
/mnt/c/Users/Wouter/Documents/code/nestmc/feature/271/build/miniapp/miniapp.exe -f 

# Do a comparison
python compare.py data/rgraph.gdf spikes_0.gdf 0.00001

# set if we have a failure in the comparison
if [ $? != 0 ]; then
    RGRAPH=1
fi

echo "running alltoall regression test"
/mnt/c/Users/Wouter/Documents/code/nestmc/feature/271/build/miniapp/miniapp.exe -f -m
python compare.py data/alltoall.gdf spikes_0.gdf 0.00001
if [ $? != 0 ]; then
    ALLTOALL=1
fi

echo "running ring regression test"
/mnt/c/Users/Wouter/Documents/code/nestmc/feature/271/build/miniapp/miniapp.exe -f -r
python compare.py data/ring.gdf spikes_0.gdf 0.00001
if [ $? != 0 ]; then
    RING=1
fi

###########################################################
# Display which test failed: It might be out of view due to the simulator output
if [ $RGRAPH ]; then
    echo "The rgraph test failed!"
    TEST_FAILED=1
fi
if [ $ALLTOALL ]; then
    echo "The all to all test failed!"
    TEST_FAILED=1
fi
if [ $RING ]; then
    echo "The ring test failed!"
    TEST_FAILED=1
fi

###########################################
# If any test failed exit with 1 exit value
if [ $TEST_FAILED ]; then
    exit 1
fi
