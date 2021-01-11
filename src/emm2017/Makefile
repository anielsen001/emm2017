CC = gcc
CFLAGS = -g -O2 -Wall -W
LDFLAGS =
LIBS = -lm

all: emm_file emm_grid emm_point

emm_file: emm_sph_file.o GeomagnetismLibrary.o
	${CC} ${LDFLAGS} -o emm_file emm_sph_file.o GeomagnetismLibrary.o ${LIBS}

emm_grid: emm_sph_grid.o GeomagnetismLibrary.o
	${CC} ${LDFLAGS} -o emm_grid emm_sph_grid.o GeomagnetismLibrary.o ${LIBS}

emm_point: emm_sph_point.o GeomagnetismLibrary.o
	${CC} ${LDFLAGS} -o emm_point emm_sph_point.o GeomagnetismLibrary.o ${LIBS}

GeomagnetismLibrary.o:
	${CC} ${CFLAGS} -c GeomagnetismLibrary.c

%.o: %.c
	${CC} ${CFLAGS} -c $<

clean:
	rm -f *.o
	rm -f emm_file emm_grid emm_point
