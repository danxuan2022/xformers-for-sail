#!/bin/bash

function compile() {
  rm -f *.o ${APP_NAME} ${APP_NAME}.${PLATFORM}
  if [ $# == 1 ] && [ $1 == 'clang']; then
    make -f ../common/makefile.clang 
  else
    make -f ../common/makefile.nvcc
  fi
  rm -f *.o
}

function run() {
  if [ -f ${APP_NAME}.${PLATFORM} ]; then
    export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:$LD_LIBRARY_PATH
    export CUDA_VISIBLE_DEVICES=0
    if [ ${PLATFORM}=='ppu' ]; then
      export LD_LIBRARY_PATH=${PPU_SDK}/lib:$LD_LIBRARY_PATH
    fi
    ./${APP_NAME}.${PLATFORM}
  else
    echo "${APP_NAME}.${PLATFORM} not exist !"
  fi
}

if [ $1 ]; then
  export APP_NAME=$1
else
  echo "Please run with './build_and_run.sh APP_NAME ppu|gpu' "
  exit
fi

PWD=`pwd`
export CUTLASS_ROOT=${PWD}/../../
# set $1=ppu or cuda
if [ $2 ] && [ $2 == "ppu" ]; then
  export PLATFORM=ppu
  export CUDA_ROOT=${PPU_SDK}/CUDA_SDK
  if [ ${PPU_SDK} ]; then 
    if [ $3 ] && [ $3 == "clang" ]; then
      echo "start to compiling based on ppu_sdk with clang" 
      compile clang
    else
      echo "start to compiling based on ppu_sdk with nvcc"
      compile
    fi
    #echo "start to run"
    #run
  else
    echo "Please set Env PPU_SDK"
  fi
elif [ $2 ] && [ $2 == "gpu" ]; then
  export PLATFORM=gpu
  export CUDA_ROOT=/usr/local/cuda
  if [ ${CUDA_ROOT} ]; then 
    echo "start to compiling with nvcc"
    compile
    #echo "start to run"
    #run
  else
    echo "Please set Env CUDA_ROOT"
  fi
else
  export PLATFORM=ppu
  export CUDA_ROOT=${PPU_SDK}/CUDA_SDK
  if [ ${PPU_SDK} ]; then
    echo "start to compiling based on ppu_sdk with nvcc"
    compile
    run
  else
    echo "Please set Env PPU_SDK"
  fi
fi

