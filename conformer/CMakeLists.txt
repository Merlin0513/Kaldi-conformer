# generated with cmake/gen_cmake_skeleton.py, DO NOT MODIFY.
set(PUBLIC_HEADERS
    conformer-activation.h
    conformer-attention.h
    conformer-convolution.h
    conformer-embedding.h
    conformer-encoder.h
    conformer-feedforward.h
    conformer-models.h
    conformer-utils.h
)

add_library(kaldi-conformer INTERFACE)
target_include_directories(kaldi-conformer INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include/kaldi>
)


install(TARGETS kaldi-conformer EXPORT kaldi-targets)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/conformer COMPONENT kaldi)

