include choose_mrbuild.mk
include $(MRBUILD_MK)/Makefile.common.header

PROJECT_NAME := dv4l
ABI_VERSION  := 0
TAIL_VERSION := 0

VERSION := $(VERSION_FROM_PROJECT)

LIB_SOURCES += \
  dv4l.c

BIN_SOURCES += \
  test.c

LDLIBS += -lavcodec -lavformat -lavutil -lswscale

CFLAGS    += --std=gnu99
CCXXFLAGS += -Wno-missing-field-initializers -Wno-unused-variable -Wno-unused-parameter

######### python stuff
dv4l-pywrap.o: $(addsuffix .h,$(wildcard *.docstring))
dv4l$(PY_EXT_SUFFIX): dv4l-pywrap.o libdv4l.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $(LDFLAGS) $(PY_MRBUILD_LDFLAGS) $< -ldv4l -o $@

PYTHON_OBJECTS := dv4l-pywrap.o

$(PYTHON_OBJECTS): CFLAGS += $(PY_MRBUILD_CFLAGS)

DIST_PY3_MODULES := dv4l

all: dv4l$(PY_EXT_SUFFIX)

include $(MRBUILD_MK)/Makefile.common.footer
