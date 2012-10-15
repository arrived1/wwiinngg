################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= wing

# Cuda source files (compiled with cudacc)
CUFILES		:= galaxy_kernel.cu kernel.cu

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= galaxy.cpp ParticleRenderer.cpp

USEGLLIB	:= 1
USEPARAMGL	:= 1
USEGLUT		:= 1

################################################################################
# Rules and targets

include ../../common/common.mk
