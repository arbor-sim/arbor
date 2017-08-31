if [[ -z "${WITH_THREAD}" ]]; then
    export WITH_THREAD=cthread
fi

if [[ -z "${WITH_DISTRIBUTED}" ]]; then
    export WITH_DISTRIBUTED=serial
fi
