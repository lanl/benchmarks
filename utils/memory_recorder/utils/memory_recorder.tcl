#%Module
# vi:set filetype=tcl:

source /usr/projects/hpcsoft/utilities/lib/envmodules_header

# Determine what MPI library we're using
if { [info exists ::env(LCOMPILER)] } {
    set compstr    $::env(LCOMPILER)
} elseif { [info exists ::env(LMOD_FAMILY_COMPILER)] } {
    set compstr    $::env(LMOD_FAMILY_COMPILER)
} else {
    set compstr    gcc
}
# Determine what MPI library we're using
if { [info exists ::env(LMPI)] } {
    set mpistr    $::env(LMPI)
} elseif { [info exists ::env(LMOD_FAMILY_MPI)] } {
    set mpistr    $::env(LMOD_FAMILY_MPI)
} else {
    set mpistr
}

set softdir      $::env(HOME)/proj/installs
set name         memory_recorder
set machine      [machineName]
set prefix       ${softdir}/${machine}/${name}/${compstr}_${mpistr}
set bindir       ${prefix}/bin
set libdir       ${prefix}/lib

setenv MEMREC_ROOT $prefix
prepend-path PATH $bindir
prepend-path LD_LIBRARY_PATH $libdir

source /usr/projects/hpcsoft/utilities/lib/envmodules_footer
