# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

project('nixlbench', 'CPP', version: '0.3.0',
    default_options: ['buildtype=release',
                'werror=true',
                'cpp_std=c++17',
                'prefix=/usr/local/nixlbench'],
    meson_version: '>= 0.64.0'
)

# set up some global vars for compiler, platform, configuration, etc.
cpp = meson.get_compiler('cpp')

# Allow overriding paths through environment variables
# CUDA
cuda_inc_path = get_option('cudapath_inc')
cuda_lib_path = get_option('cudapath_lib')
cuda_stub_path = get_option('cudapath_stub')
# ETCD
etcd_inc_path = get_option('etcd_inc_path')
etcd_lib_path = get_option('etcd_lib_path')
# NIXL
nixl_path = get_option('nixl_path')

# Find required libraries
# NIXL
host_cpu_family = host_machine.cpu_family()
host_system = host_machine.system().to_lower()

if host_system != 'linux' or host_cpu_family not in ['x86_64', 'aarch64']
    error('This build only supports Linux on x86_64 or aarch64 architectures.')
endif

nixl_lib_path = nixl_path + '/lib/' + host_cpu_family + '-linux-gnu'
nixl_lib = cpp.find_library('nixl', dirs: [nixl_lib_path])
nixl_build = cpp.find_library('nixl_build', dirs: [nixl_lib_path])
nixl_serdes = cpp.find_library('serdes', dirs: [nixl_lib_path])

if not nixl_lib.found() or not nixl_build.found() or not nixl_serdes.found()
    error('NIXL Libraries not found. Exiting.')
endif

# CUDA
cuda_available = false
if cuda_lib_path == ''
    cuda_dep = dependency('cuda', required : false, modules : [ 'cudart', 'cuda' ])
    if cuda_dep.found()
        cuda_available = true
    endif
else
    message('cuda lib path ', cuda_lib_path)
    if cuda_stub_path == ''
        cuda_stub_path = cuda_lib_path + '/stubs'
    endif
    cuda_dep = declare_dependency(
        link_args : ['-L' + cuda_lib_path, '-L' + cuda_stub_path, '-lcuda', '-lcudart'],
        include_directories : include_directories(cuda_inc_path))
    cuda_available = true
endif

# UCX
ucx_dep = dependency('ucx')

# GFlags
gflags_dep = dependency('gflags', required: true)

# OpenMP
openmp_dep = dependency('openmp', required: true)

# Check for etcd-cpp-api - use multiple methods for discovery
etcd_dep = dependency('etcd-cpp-api', required : false)

# Ensure etcd is available
etcd_available = etcd_dep.found()
if etcd_available
    add_project_arguments('-DHAVE_ETCD', language: 'cpp')
else
    message('ETCD C++ client library not found. Disabling ETCD runtime.')
endif

etcd_inc = etcd_inc_path
nixl_inc = nixl_path + '/include'

inc_dir = include_directories('.', './src/', nixl_inc)

if cuda_available
    add_project_arguments('-DHAVE_CUDA', language: 'cpp')
endif

# Subprojects
subdir('src/utils')
subdir('src/runtime')
subdir('src/worker')

# Configure header file
configure_file(
    output: 'config.h',
    configuration: {
        'HAVE_ETCD': etcd_available ? '1' : '0',
        'HAVE_CUDA': cuda_available ? '1' : '0',
    },
    install: true,
    install_dir: get_option('includedir') / 'nixlbench'
)

deps = [gflags_dep, nixl_lib, nixl_build, nixl_serdes, openmp_dep]
args = []
if etcd_available
    deps += [etcd_dep]
endif
if cuda_available
    deps += [cuda_dep]
endif

if not etcd_available
    error('No runtime available or not found')
endif


executable('nixlbench', 'src/main.cpp',
            include_directories: inc_dir,
            link_with: [nixlbench_runtimes, utils_lib, worker_libs],
            dependencies: deps,
            link_args: args,
            install: true,
            install_dir: get_option('bindir'))

executable('p2p_transfer', 'src/p2p_main.cpp',
            include_directories: inc_dir,
            link_with: [nixlbench_runtimes, utils_lib, worker_libs],
            dependencies: deps,
            link_args: args,
            install: true,
            install_dir: get_option('bindir'))