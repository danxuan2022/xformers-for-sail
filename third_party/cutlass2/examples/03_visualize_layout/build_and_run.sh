#!/bin/bash

function compile() {
  rm -f *.o ${APP_NAME} ${APP_NAME}.${PLATFORM}
  if [ $# == 1 ] && [ $1 == 'clang']; then
    make -f makefile.clang 
  else
    make -f makefile.nvcc
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
    ./${APP_NAME}.${PLATFORM} "TensorOpMultiplicand<32,16>" --extent=16,32 --vectorize=4 --output-shape=32,4  
  else
    echo "${APP_NAME}.${PLATFORM} not exist !"
  fi
}

export APP_NAME=visualize_layout
PWD=`pwd`
export CUTLASS_ROOT=${PWD}/../../
# set $1=ppu or cuda
if [ $1 ] && [ $1 == "ppu" ]; then
  export PLATFORM=ppu
  export CUDA_ROOT=${PPU_SDK}/CUDA_SDK
  if [ ${PPU_SDK} ]; then 
    if [ $2 ] && [ $2 == "clang" ]; then
      echo "start to compiling with clang" 
      compile clang
    else
      echo "start to compiling with nvcc"
      compile
    fi
    echo "start to run"
    run
  else
    echo "Please set Env PPU_SDK"
  fi
elif [ $1 ] && [ $1 == "gpu" ]; then
  export PLATFORM=gpu
  if [ ${CUDA_ROOT} ]; then 
    echo "start to compiling with nvcc"
    compile
    echo "start to run"
    run
  else
    echo "Please set Env CUDA_ROOT"
  fi
else
  echo "Please run with './build_and_run.sh ppu|gpu'"	
fi

