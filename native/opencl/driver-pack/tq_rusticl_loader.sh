#!/system/bin/sh
# TurboQuant Rusticl loader for KGSL/Adreno
export LD_LIBRARY_PATH=/data/data/com.termux/files/tq-rusticl:$LD_LIBRARY_PATH
export OCL_ICD_VENDORS=/data/data/com.termux/files/tq-rusticl
export TQ_DRIVER_ROOT=/data/data/com.termux/files/tq-rusticl
export MESA_NO_ERROR=1
exec /data/data/com.termux/files/tq-rusticl/libRusticlOpenCL.so.1.0.0 "$@"