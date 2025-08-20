#
# Copyright 2023 Ben Worpitz, Jan Stephan
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.25)

project("alpakaExamples" LANGUAGES CXX)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory("grayScale/")
add_subdirectory("heatEquation2D/")
add_subdirectory("scan/")
add_subdirectory("tensor/tensorBasicExample/")
add_subdirectory("tensor/tensorFunctionality/")
add_subdirectory("tensor/tensorPerformanceComparison/")
add_subdirectory("tensor/tensorSyncTest/")
add_subdirectory("tutorial/")
add_subdirectory("vectorAdd/")
