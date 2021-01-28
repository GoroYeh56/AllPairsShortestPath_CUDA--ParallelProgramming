NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
LDFLAGS  := -lm
EXES     := hw4-2 #Verification_twoFiles
CXXFLAGS := -fopenmp

# TARGETS := nvprof 
# CASE = 04.1
# # OUTPUT_FILEPATH = p$(CASE).nvvp
# prof-metrics: $(EXES)
# 	srun -n 1 -c 2 --gres=gpu:2 -pprof nvprof --profile-from-start off \
# 	--metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,dram_read_throughput,dram_write_throughput \
# 	./hw4-2 cases/c$(CASE).in c$(CASE).out


alls: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

hw4-2: hw4-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -Xcompiler=$(CXXFLAGS) -o $@ $?
