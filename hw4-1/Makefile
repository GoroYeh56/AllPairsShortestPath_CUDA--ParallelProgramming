NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
LDFLAGS  := -lm #--no-relax
EXES     := hw4-1 #Verification_twoFiles

# CC = mpicc
# CXX = mpicxx
CXXFLAGS = -O3 -lm
CFLAGS = -O3 -lm

# TARGETS := nvprof 
# CASE = 11k1
# OUTPUT_FILEPATH = p$(CASE).nvvp
# prof-metrics: $(EXES)
# 	srun -n 1 --gres=gpu:1 -pprof nvprof --profile-from-start off \
# 	--metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,dram_read_throughput,dram_write_throughput \
# 	--log-file $(OUTPUT_FILEPATH) \
# 	./hw4-1 cases/p$(CASE) mycases/p$(CASE).out

# TARGETS := nvprof 
# CASE = 11k1
# OUTPUT_FILEPATH = p$(CASE).nvvp
# prof-metrics: $(EXES)
# 	srun -n 1 --gres=gpu:1 -pprof nvprof --profile-from-start off \
# 	--events shared_ld_bank_conflict,shared_st_bank_conflict \
# 	--metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,dram_read_throughput,dram_write_throughput \
# 	./hw4-1 cases/p$(CASE) mycases/p$(CASE).out

# TARGETS := nvprof 
# CASE = 11k1
# OUTPUT_FILEPATH = p$(CASE).nvvp
# prof-metrics: $(EXES)
# 	srun -n 1 --gres=gpu:1 -pprof nvprof --profile-from-start off \
# 	--metrics inst_integer,gld_throughput,gst_throughput \
# 	./hw4-1 cases/p$(CASE) mycases/p$(CASE).out


alls: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

Verification_twoFiles: Verification_twoFiles.cpp
	g++ $(CXXFLAGS) -o $@ $?

hw4-1: hw4-1.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?


# prof-metrics: $(TARGETS)
# 	srun -n 1 --gres=gpu:1 -pprof nvprof --profile-from-start off \
# 	--metrics achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,dram_read_throughput,dram_write_throughput \
# 	./hw4-1 cases/p$(CASE) mycases/p$(CASE).out