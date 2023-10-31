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

echo "GNUPLOT INFO: execution within ` pwd -P `"
echo "GNUPLOT INFO: gnuplot: ` which gnuplot `"
echo "GNUPLOT INFO: find: ` which find `"
echo "GNUPLOT INFO: xargs: ` which xargs `"
echo "GNUPLOT INFO: dirname: ` which dirname `"
echo "GNUPLOT INFO: basename: ` which basename `"

find .. -not \( -path "./archive" -prune \) -name "*.gp" -print0 |  xargs -0 -I file bash -c "do_single file"

exit 0
