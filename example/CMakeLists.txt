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
add_subdirectory("tutorial/")
add_subdirectory("vectorAdd/")
add_subdirectory("tensorFunctionality/")
add_subdirectory("tensorBasicExample/")
add_subdirectory("tensorAsync/")
add_subdirectory("tensorSyncTest2/")
