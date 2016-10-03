prefix=$(dirname $0)
for f in `ls $prefix/modfiles/*.mod`
do
    for target in cpu gpu
    do
        logfile=$f.$target.log
        printf "testing %30s::%4s : " $f $target
        valgrind --leak-check=full $prefix/../bin/modcc $f -t $target -o tmp.h &> $logfile
        awk 'BEGIN { err = 0 } \
/ERROR SUMMARY/   { if ($4 != 0) ++err } \
/definitely lost/ { if ($4 != 0) ++err } \
END { if (err) print("fail"); else print("success") }' $logfile
    done
done
