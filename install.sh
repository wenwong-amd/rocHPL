#!/usr/bin/env bash
# Author: Nico Trost
# Modified by: Noel Chalmers

#set -x #echo on

# #################################################
# helper functions
# #################################################
function display_help()
{
  echo "rocHPL build helper script"
  echo "./install "
  echo "    [-h|--help] prints this help message"
  echo "    [-i|--install] install after build"
  echo "    [-d|--dependencies] install dependencies"
  echo "    [-g|--debug] Set build type to Debug (otherwise build Release)"
  echo "    [--with-rocm=<dir>] Path to ROCm install (Default: /opt/rocm)"
  echo "    [--with-rocblas=<dir>] Path to rocBLAS library (Default: /opt/rocm/rocblas)"
  echo "    [--with-cpublas=<dir>] Path to external CPU BLAS library (Default: clone+build BLIS in tpl/)"
  echo "    [--with-mpi=<dir>] Path to external MPI install (Default: clone+build OpenMPI v4.0.5 in tpl/)"
  echo "    [--gpu-aware-mpi] MPI library supports GPU-aware communication (Default: false)"
  echo "    [--verbose-print] Verbose output during HPL setup (Default: true)"
  echo "    [--progress-report] Print progress report to terminal during HPL run (Default: true)"
  echo "    [--detailed-timing] Record detailed timers during HPL run (Default: true)"
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
# true is a system command that completes successfully, function returns success
# prereq: ${ID} must be defined before calling
supported_distro( )
{
  if [ -z ${ID+foo} ]; then
    printf "supported_distro(): \$ID must be set\n"
    exit 2
  fi

  case "${ID}" in
    ubuntu|centos|rhel|fedora|sles)
        true
        ;;
    *)  printf "This script is currently supported on Ubuntu, CentOS, RHEL, Fedora and SLES\n"
        exit 2
        ;;
  esac
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
check_exit_code( )
{
  if (( $? != 0 )); then
    exit $?
  fi
}

# This function is helpful for dockerfiles that do not have sudo installed, but the default user is root
elevate_if_not_root( )
{
  local uid=$(id -u)

  if (( ${uid} )); then
    sudo $@
    check_exit_code
  else
    $@
    check_exit_code
  fi
}

# Take an array of packages as input, and install those packages with 'apt' if they are not already installed
install_apt_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dpkg-query --show --showformat='${db:Status-Abbrev}\n' ${package} 2> /dev/null | grep -q "ii"; echo $?) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root apt install -y --no-install-recommends ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'yum' if they are not already installed
install_yum_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(yum list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root yum -y --nogpgcheck install ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'dnf' if they are not already installed
install_dnf_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(dnf list installed ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root dnf install -y ${package}
    fi
  done
}

# Take an array of packages as input, and install those packages with 'zypper' if they are not already installed
install_zypper_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    if [[ $(rpm -q ${package} &> /dev/null; echo $? ) -ne 0 ]]; then
      printf "\033[32mInstalling \033[33m${package}\033[32m from distro package manager\033[0m\n"
      elevate_if_not_root zypper -n --no-gpg-checks install ${package}
    fi
  done
}

# Take an array of packages as input, and delegate the work to the appropriate distro installer
# prereq: ${ID} must be defined before calling
install_packages( )
{
  if [ -z ${ID+foo} ]; then
    printf "install_packages(): \$ID must be set\n"
    exit 2
  fi

  # dependencies needed for executable to build
  local library_dependencies_ubuntu=( "make" "cmake" "libnuma-dev" "pkg-config" "autoconf" "libtool" "automake" "m4" "flex" )
  local library_dependencies_centos=( "make" "cmake3" "gcc-c++" "rpm-build" "epel-release" "numactl-libs" "autoconf" "libtool" "automake" "m4" "flex" )
  local library_dependencies_fedora=( "make" "cmake" "gcc-c++" "libcxx-devel" "rpm-build" "numactl-libs"  "autoconf" "libtool" "automake" "m4" "flex" )
  local library_dependencies_sles=(   "make" "cmake" "gcc-c++" "libcxxtools9" "rpm-build" "libnuma-devel" "autoconf" "libtool" "automake" "m4" "flex" )

  if [[ "${with_rocm}" == /opt/rocm ]]; then
    library_dependencies_ubuntu+=("rocblas" "rocblas-dev")
    library_dependencies_centos+=("rocblas" "rocblas-devel")
    library_dependencies_fedora+=("rocblas" "rocblas-dev")
    library_dependencies_sles+=("rocblas" "rocblas-devel")
  fi

  case "${ID}" in
    ubuntu)
      elevate_if_not_root apt update
      install_apt_packages "${library_dependencies_ubuntu[@]}"

      ;;

    centos|rhel)
#     yum -y update brings *all* installed packages up to date
#     without seeking user approval
#     elevate_if_not_root yum -y update
      install_yum_packages "${library_dependencies_centos[@]}"

      ;;

    fedora)
#     elevate_if_not_root dnf -y update
      install_dnf_packages "${library_dependencies_fedora[@]}"

      ;;

    sles)
#     elevate_if_not_root zypper -y update
      install_zypper_packages "${library_dependencies_sles[@]}"

       ;;
    *)
      echo "This script is currently supported on Ubuntu, CentOS, RHEL, Fedora and SLES"
      exit 2
      ;;
  esac
}

check_apt_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    dpkg-query -W $package > /dev/null
    if [ $? -eq 1 ]; then
      printf "\033[31mRequired package \033[33m${package}\033[31m is not installed.\033[0m\n"
      printf "\033[31mPlease install required package or disable corresponding HPCG build option.\033[0m\n"
      exit 2
    fi
  done
}

check_yum_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    rpm -q $package > /dev/null
    if [ $? -eq 1 ]; then
      printf "\033[31mRequired package \033[33m${package}\033[31m is not installed.\033[0m\n"
      printf "\033[31mPlease install required package or disable corresponding HPCG build option.\033[0m\n"
      exit 2
    fi
  done
}

check_dnf_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    rpm -q $package > /dev/null
    if [ $? -eq 1 ]; then
      printf "\033[31mRequired package \033[33m${package}\033[31m is not installed.\033[0m\n"
      printf "\033[31mPlease install required package or disable corresponding HPCG build option.\033[0m\n"
      exit 2
    fi
  done
}

check_zypper_packages( )
{
  package_dependencies=("$@")
  for package in "${package_dependencies[@]}"; do
    rpm -q $package > /dev/null
    if [ $? -eq 1 ]; then
      printf "\033[31mRequired package \033[33m${package}\033[31m is not installed.\033[0m\n"
      printf "\033[31mPlease install required package or disable corresponding HPCG build option.\033[0m\n"
      exit 2
    fi
  done
}

check_packages( )
{
  if [ -z ${ID+foo} ]; then
    printf "check_packages(): \$ID must be set\n"
    exit 2
  fi

  package_dependency=("$@")

  if [[ "$package_dependency" == omp ]]; then
    local build_dependencies_ubuntu=( "libomp-dev" )
    local build_dependencies_centos=( "libgomp" )
    local build_dependencies_fedora=( "libgomp" )
    local build_dependencies_sles=( "libgomp1" )
  fi

  case "${ID}" in
    ubuntu)
      check_apt_packages "${build_dependencies_ubuntu[@]}"
      ;;
    centos|rhel)
      check_yum_packages "${build_dependencies_centos[@]}"
      ;;
    fedora)
      check_dnf_packages "${build_dependencies_fedora[@]}"
      ;;
    sles)
      check_zypper_packages "${build_dependencies_sles[@]}"
      ;;
    *)
      echo "This script is currently supported on Ubuntu, CentOS, RHEL and Fedora"
      exit 2
      ;;
  esac
}

# Install BLIS in rochpl/tpl
install_blis( )
{
  if [ ! -d "./tpl/blis" ]; then
    mkdir -p tpl && cd tpl
    git clone https://github.com/amd/blis --branch 3.0.1
    cd blis; ./configure --prefix=${PWD} --enable-cblas auto;
    make -j$(nproc); make install -j$(nproc); cd ../..
  elif [ ! -f "./tpl/blis/lib/libblis.so" ]; then
    cd tpl/blis; ./configure --prefix=${PWD} --enable-cblas auto;
    make -j$(nproc); make install -j$(nproc); cd ../..
  fi

  # Check for successful build
  if [ ! -f "./tpl/blis/lib/libblis.so" ]; then
    echo "Error: BLIS install unsuccessful."
    exit 3
  fi
}

# Clone and build OpenMPI+UCX in rochpl/tpl
install_openmpi( )
{
  ucx_lib_folder=./tpl/ucx/lib
  ompi_lib_folder=./tpl/openmpi/lib
  case "${ID}" in
    sles)
      ucx_lib_folder=./tpl/ucx/lib64
      ompi_lib_folder=./tpl/openmpi/lib64
      ;;
  esac

  if [ ! -d "./tpl/ucx" ]; then
    mkdir -p tpl && cd tpl
    git clone --branch v1.11.2 https://github.com/openucx/ucx.git ucx
    cd ucx; ./autogen.sh; ./autogen.sh #why do we have to run this twice?
    mkdir build; cd build
    ../contrib/configure-opt --prefix=${PWD}/../ --with-rocm=${with_rocm} --without-knem --without-cuda --without-java
    make -j$(nproc); make install; cd ../../..
  elif [ ! -f "${ucx_lib_folder}/libucm.so" ] || [ ! -f "${ucx_lib_folder}/libucp.so" ] || \
       [ ! -f "${ucx_lib_folder}/libucs.so" ] || [ ! -f "${ucx_lib_folder}/libuct.so" ]; then
    cd tpl/ucx; ./autogen.sh; ./autogen.sh
    mkdir build; cd build
    ../contrib/configure-opt --prefix=${PWD}/../ --with-rocm=${with_rocm} --without-knem --without-cuda --without-java
    make -j$(nproc); make install; cd ../../..
  fi

  # Check for successful build
  if [ ! -f "${ucx_lib_folder}/libucm.so" ] || [ ! -f "${ucx_lib_folder}/libucp.so" ] || \
     [ ! -f "${ucx_lib_folder}/libucs.so" ] || [ ! -f "${ucx_lib_folder}/libuct.so" ]; then
    echo "Error: UCX install unsuccessful."
    exit 3
  fi

  if [ ! -d "./tpl/openmpi" ]; then
    mkdir -p tpl && cd tpl
    git clone --branch v4.1.1 https://github.com/open-mpi/ompi.git openmpi
    cd openmpi; ./autogen.pl; mkdir build; cd build
    ../configure --prefix=${PWD}/../ --with-ucx=${PWD}/../../ucx --without-verbs
    make -j$(nproc); make install; cd ../../..
  elif [ ! -f "${ompi_lib_folder}/libmpi.so" ]; then
    cd tpl/openmpi; ./autogen.pl; mkdir build; cd build
    ../configure --prefix=${PWD}/../ --with-ucx=${PWD}/../../ucx --without-verbs
    make -j$(nproc); make install; cd ../../..
  fi

  # Check for successful build
  if [ ! -f "${ompi_lib_folder}/libmpi.so" ]; then
    echo "Error: OpenMPI install unsuccessful."
    exit 3
  fi
}

# #################################################
# Pre-requisites check
# #################################################
# Exit code 0: alls well
# Exit code 1: problems with getopt
# Exit code 2: problems with supported platforms

# check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
  echo "This script uses getopt to parse arguments; try installing the util-linux package";
  exit 1
fi

# os-release file describes the system
if [[ -e "/etc/os-release" ]]; then
  source /etc/os-release
else
  echo "This script depends on the /etc/os-release file"
  exit 2
fi

# The following function exits script if an unsupported distro is detected
supported_distro

# #################################################
# global variables
# #################################################
install_package=false
install_dependencies=false
install_prefix=rochpl-install
build_release=true
with_rocm=/opt/rocm
with_mpi=tpl/openmpi
with_rocblas=/opt/rocm/rocblas
with_cpublas=tpl/blis/lib
gpu_aware_mpi=OFF
openmpi_ucx=false
verbose_print=true
progress_report=true
detailed_timing=true

# #################################################
# Parameter parsing
# #################################################

# check if we have a modern version of getopt that can handle whitespace and long parameters
getopt -T
if [[ $? -eq 4 ]]; then
  GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,install,dependencies,debug,with-rocm:,with-mpi:,with-rocblas:,with-cpublas:,gpu-aware-mpi:,verbose-print:,progress-report:,detailed-timing: --options hidg -- "$@")
else
  echo "Need a new version of getopt"
  exit 1
fi

if [[ $? -ne 0 ]]; then
  echo "getopt invocation failed; could not parse the command line";
  exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
  case "${1}" in
    -h|--help)
        display_help
        exit 0
        ;;
    -i|--install)
        install_package=true
        shift ;;
    -d|--dependencies)
        install_dependencies=true
        shift ;;
    -g|--debug)
        build_release=false
        shift ;;
    --with-rocm)
        with_rocm=${2}
        shift 2 ;;
    --with-mpi)
        with_mpi=${2}
        shift 2 ;;
    --with-rocblas)
        with_rocblas=${2}
        shift 2 ;;
    --with-cpublas)
        with_cpublas=${2}
        shift 2 ;;
    --gpu-aware-mpi)
        gpu_aware_mpi=${2}
        shift 2 ;;
    --verbose-print)
        verbose_print=${2}
        shift 2 ;;
    --progress-report)
        progress_report=${2}
        shift 2 ;;
    --detailed-timing)
        detailed_timing=${2}
        shift 2 ;;
    --) shift ; break ;;
    *)  echo "Unexpected command line parameter received; aborting";
        exit 1
        ;;
  esac
done

build_dir=./build
printf "\033[32mCreating project build directory in: \033[33m${build_dir}\033[0m\n"

# #################################################
# prep
# #################################################
# ensure a clean build environment
rm -rf ${build_dir}

# Default cmake executable is called cmake
cmake_executable=cmake

case "${ID}" in
  centos|rhel)
  cmake_executable=cmake3
  ;;
esac

# #################################################
# dependencies
# #################################################
if [[ "${install_dependencies}" == true ]]; then

  install_packages

fi

# We append customary rocm path; if user provides custom rocm path in ${path}, our
# hard-coded path has lesser priority
export ROCM_PATH=${with_rocm}
export PATH=${PATH}:${ROCM_PATH}/bin

pushd .
  # #################################################
  # BLAS
  # #################################################
  if [[ "${with_cpublas}" == tpl/blis/lib ]]; then

    install_blis

  fi

  # #################################################
  # MPI
  # #################################################
  if [[ "${with_mpi}" == tpl/openmpi ]]; then

    #gpu_aware_mpi=ON #turn on GPU-aware MPI when using internal MPI library
    with_mpi=${PWD}/tpl/openmpi
    openmpi_ucx=true
    install_openmpi

  fi

  # #################################################
  # configure & build
  # #################################################
  cmake_common_options="-DCMAKE_INSTALL_PREFIX=${install_prefix} -DHPL_BLAS_DIR=${with_cpublas}
                        -DHPL_MPI_DIR=${with_mpi} -DROCM_PATH=${with_rocm} -DROCBLAS_PATH=${with_rocblas}"

  # build type
  cmake_common_options="${cmake_common_options} -DCMAKE_BUILD_TYPE=Release"

  shopt -s nocasematch
  if [[ "${gpu_aware_mpi}" == on || "${gpu_aware_mpi}" == true || "${gpu_aware_mpi}" == 1 || "${gpu_aware_mpi}" == enabled ]]; then
    cmake_common_options="${cmake_common_options} -DGPU_AWARE_MPI=ON"
  fi
  if [[ "${verbose_print}" == on || "${verbose_print}" == true || "${verbose_print}" == 1 || "${verbose_print}" == enabled ]]; then
    cmake_common_options="${cmake_common_options} -DHPL_VERBOSE_PRINT=ON"
  fi
  if [[ "${progress_report}" == on || "${progress_report}" == true || "${progress_report}" == 1 || "${progress_report}" == enabled ]]; then
    cmake_common_options="${cmake_common_options} -DHPL_PROGRESS_REPORT=ON"
  fi
  if [[ "${detailed_timing}" == on || "${detailed_timing}" == true || "${detailed_timing}" == 1 || "${detailed_timing}" == enabled ]]; then
    cmake_common_options="${cmake_common_options} -DHPL_DETAILED_TIMING=ON"
  fi
  shopt -u nocasematch

  if [[ "${openmpi_ucx}" == true ]]; then
    cmake_common_options="${cmake_common_options} -DHPL_OPENMPI_UCX=ON"
  fi

  # Build library with AMD toolchain because of existense of device kernels
  mkdir -p ${build_dir} && cd ${build_dir}
  ${cmake_executable} ${cmake_common_options} ..
  check_exit_code

  make -j$(nproc) install
  check_exit_code

  # #################################################
  # install
  # #################################################
  # installing through package manager, which makes uninstalling easy
  if [[ "${install_package}" == true ]]; then
    make package
    check_exit_code

    case "${ID}" in
      ubuntu)
        elevate_if_not_root dpkg -i rochpl*.deb
      ;;
      centos|rhel)
        elevate_if_not_root yum -y localinstall rochpl*.rpm
      ;;
      fedora)
        elevate_if_not_root dnf install rochpl*.rpm
      ;;
      sles)
        elevate_if_not_root zypper -n --no-gpg-checks install rochpl*.rpm
      ;;
    esac
  fi
popd
