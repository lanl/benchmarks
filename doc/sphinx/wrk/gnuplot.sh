#!/bin/sh

export APP_GNUPLOT="${APP_GNUPLOT:-gnuplot}"

do_single()
{
    m_dir=`dirname "${1}"`
    m_gp=`basename "${1}"`
    cd ${m_dir}
    set -x
    "${APP_GNUPLOT}" "${m_gp}"
    set +x
}
export -f do_single

find .. -name "*.gp" -print0 \
    | xargs -0 -I file bash -c "do_single file"

exit 0
