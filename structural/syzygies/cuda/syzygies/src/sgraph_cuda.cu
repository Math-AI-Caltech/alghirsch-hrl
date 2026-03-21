/**
 * @file sgraph.cu
 * @brief Implements kernels.
 */

#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>

#include "sgraph.hpp"

namespace syzygy {

// #define SGRAPH_DEBUG_

// KERNELS

__global__
void fill_row_ids(
        const int64_t* __restrict__ row_ptr,
        const int64_t* __restrict__ col_ids,
        int64_t* __restrict__ row_ids,
        int64_t num_rows) {
    // CSR -> COO to avoid recomputation of sources when selecting edges.
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_rows) return;

    for (int64_t j = row_ptr[i]; j < row_ptr[i+1]; ++j) row_ids[j] = i;
}

__device__
monom_t weight(int64_t s_idx, int64_t t_idx, const monom_t* __restrict__ generators) {
    return generators[t_idx] & (~generators[s_idx]);
}

__global__
void reduce_kernel(
        const int64_t* __restrict__ row_ptr, // CSR
        const int64_t* __restrict__ col_ids, // CSR | COO
        const int64_t* __restrict__ row_ids, // COO
        const monom_t* __restrict__ generators,
        const int64_t* __restrict__ level_offsets,
        const bool* __restrict__ masked_vertices,
        int level,
        int num_nodes,
        int64_t num_edges,
        int* count,
        int* irreducible_edges) {
    // Each block is associated to an edge and runs BFS on it.
    int block_idx = blockIdx.x;
    if (block_idx >= (int)num_edges) return;

    int tid = threadIdx.x;
    if (irreducible_edges && tid == 0) {
        irreducible_edges[2*block_idx] = -1;
        irreducible_edges[2*block_idx+1] = -1;
    }


    int64_t row_idx = (row_ids + level_offsets[level-1])[block_idx];
    if (!masked_vertices[row_idx]) return;
    int64_t col_idx = (col_ids + level_offsets[level-1])[block_idx];
    if (!masked_vertices[col_idx]) return;

    if (row_idx > col_idx) return;

    // This is shared within current block, which is running BFS
    // shared_block_data = |frontier|next_frontier|visited|, each num_nodes size(*appropriate type).
    extern __shared__ int shared_block_data[];
    __shared__ int* frontier;
    __shared__ monom_t* gens;
    __shared__ int* next_frontier;
    __shared__ monom_t* next_gens;
    // TODO more memory efficient way to store visited, A100 has only ~150KB shared memory per block.
    __shared__ int* visited;

    // Thread idx runs over vertices in the frontier
    __shared__ int frontier_size;
    __shared__ int next_frontier_size;
    __shared__ int found_path;

    // Start the loop with frontier being of size 1
    if (tid == 0) {
        frontier = shared_block_data;
        gens = frontier + (int)num_nodes;
        next_frontier = gens + (int)num_nodes;
        next_gens = next_frontier + (int)num_nodes;
        // TODO more memory efficient way to store visited, A100 has only ~150KB shared memory per block.
        visited = next_gens + (int)num_nodes;
        // printf("(%lld,%lld),\n", row_idx, col_idx);
        found_path = 0;
        frontier[0] = row_idx;
        gens[0] = weight(row_idx, col_idx, generators);
        frontier_size = 1;
        next_frontier_size = 0;
    }
    __syncthreads(); // important, visited might point to null
    // make sure all nodes are not visited yet
    for (int i = tid; i < num_nodes; i += blockDim.x) visited[i] = 0;
    __syncthreads();

    while (frontier_size > 0) {
        // Parallel loop over all frontiers
        for (int f_idx = tid; f_idx < frontier_size; f_idx += blockDim.x) {
            int64_t s = frontier[f_idx];
            monom_t s_gen = gens[f_idx];
            for (int k_idx = row_ptr[s]; k_idx < row_ptr[s+1]; ++k_idx) {
                int64_t k = col_ids[k_idx];
                if (!masked_vertices[k]) continue;

                // TODO more efficient way of tracking visited
                if (atomicCAS(&visited[k], 0, 1) != 0) continue;

                if ((s_gen | weight(s, k, generators)) == s_gen) {
                    monom_t prod = s_gen & (~weight(s, k, generators));
                    if (k == col_idx) atomicExch(&found_path, 1);

                    int pos = atomicAdd(&next_frontier_size, 1);
                    next_frontier[pos] = k;
                    next_gens[pos] = (weight(k, s, generators) | prod);
                }
            }
        }

        // Swap frontiers
        __syncthreads();
        if (tid == 0) {
            int* f_tmp = frontier;
            monom_t* g_tmp = gens;

            frontier = next_frontier;
            gens = next_gens;

            next_frontier = f_tmp;
            next_gens = g_tmp;

            // Terminate loop if one of the threads finds a path.
            frontier_size = (found_path == 1) ? 0 : next_frontier_size;
            next_frontier_size = 0;
        }
        __syncthreads();
    }

    __syncthreads();
    if (found_path != 1 && tid == 0) {
        if (count) atomicAdd(count, 1);
        if (irreducible_edges) {
            irreducible_edges[2*block_idx]      = row_idx;
            irreducible_edges[2*block_idx+1]    = col_idx;
        }
    }
}

// PRIVATE

void SGraph::cuda_deallocate() {
    if (!cu_buffers_filled) return;

    cudaFree(d_all_row_ptr);
    cudaFree(d_all_col_ids);
    cudaFree(d_generators);

    cudaFree(d_all_row_ids);
    cudaFree(d_level_offsets);
}

// PUBLIC

void SGraph::fill_buffers_cu() {
    // Shared memory buffer size limit
    cudaFuncSetAttribute(reduce_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 1 << 16); // use 17 if A100 or equivalent

    // Allocate offsets
    std::vector<int64_t> level_offsets;
    level_offsets.push_back(0);
    for (int level = 1; level <= max_level; ++level)
        level_offsets.push_back(level_offsets.back() + 2*this->num_edges(level));
    cudaMalloc((void**)&d_level_offsets, sizeof(int64_t)*level_offsets.size());
    cudaMemcpy(d_level_offsets, level_offsets.data(), sizeof(int64_t)*level_offsets.size(), cudaMemcpyHostToDevice);

    // Allocate CSR and COO graph data. Memory arrangement |level1|level2|...|max_level|, where each has the same format as host.
    cudaMalloc((void**)&d_all_row_ptr, sizeof(int64_t)*max_level*(generators.size()+1));
    cudaMalloc((void**)&d_all_col_ids, sizeof(int64_t)*level_offsets.back());
    cudaMalloc((void**)&d_all_row_ids, sizeof(int64_t)*level_offsets.back());

    for (int level = 1; level <= max_level; ++level) {
        int64_t num_edges = 2*this->num_edges(level);
        cudaMemcpy(d_all_row_ptr + (level-1)*(generators.size()+1), h_all_row_ptr[level-1], sizeof(int64_t)*(generators.size()+1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_all_col_ids + (level_offsets[level-1]), h_all_col_ids[level-1], sizeof(int64_t)*num_edges, cudaMemcpyHostToDevice);

        int block_size = 256; // TODO figure out dynamical block_size;
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        fill_row_ids<<<(generators.size() + block_size - 1) / block_size, block_size, 0, stream>>>(
            d_all_row_ptr + (level-1)*(generators.size()+1),
            d_all_col_ids + level_offsets[level-1],
            d_all_row_ids + level_offsets[level-1],
            generators.size());
    }

    // Allocate generators
    cudaMalloc((void**)&d_generators, sizeof(monom_t)*generators.size());
    cudaMemcpy(d_generators, generators.data(), sizeof(monom_t)*generators.size(), cudaMemcpyHostToDevice);

    // Count
    cudaMalloc((void**)&d_count, sizeof(int));

    cudaDeviceSynchronize();
    cu_buffers_filled = true;

    #ifdef SGRAPH_DEBUG_
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("fill_buffers_cu(): CUDA error: %s\n", cudaGetErrorString(err));
    }
    #endif
}


int64_t SGraph::count_irreducible_cu(bool* masked_vertices) {
    // An altenative is to handle this in kernel by something like: if (threadIdx.x + blockIdx.x*blockDim.x == 0) *d_count = 0;
    cudaMemset(d_count, 0, sizeof(int));
    for (int level = 2; level <= this->max_level; ++level) {
        int num_nodes = this->generators.size();
        int64_t num_edges = 2*this->num_edges(level);

        // printf("allocating_block_size: %llu KB\n", sizeof(int)*num_nodes*5 / 1000);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        reduce_kernel<<<num_edges, min(num_nodes, 1024), sizeof(int)*num_nodes*5, stream>>>(
            d_all_row_ptr,   // const int64_t* __restrict__ row_ptr, // CSR
            d_all_col_ids,   // const int64_t* __restrict__ col_ids, // CSR | COO
            d_all_row_ids,   // const int64_t* __restrict__ row_ids, // COO
            d_generators,    // const monom_t* __restrict__ generators,
            d_level_offsets, // const int64_t* __restrict__ level_offsets,
            masked_vertices, // const bool* __restrict__ masked_vertices,
            level,           // int level
            num_nodes,       // int num_nodes,
            num_edges,       // int64_t num_edges
            d_count,         // int* count
            nullptr          // int* irreducible_edges
        );

        #ifdef SGRAPH_DEBUG_
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error: %s. Attempted to pass: |E|=%lld, |V|=%d\n", cudaGetErrorString(err), num_edges, min(num_nodes, 1024));
            }
        #endif
    }

    #ifdef SGRAPH_DEBUG_
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    #endif

    int count = 0;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    return static_cast<int64_t>(count);
}


void SGraph::irreducible_edges_cu(bool* masked_vertices, vector<int>& irreducible) {
    for (int level = 2; level <= this->max_level; ++level) {
        int num_nodes = this->generators.size();
        int64_t num_edges = 2*this->num_edges(level);

        // Preallocate these
        int* irr_edges_cu;
        cudaMalloc((void**)&irr_edges_cu, 2*sizeof(int)*num_edges);

        // printf("allocating_block_size: %llu KB\n", sizeof(int)*num_nodes*5 / 1000);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        reduce_kernel<<<num_edges, min(num_nodes, 1024), sizeof(int)*num_nodes*5, stream>>>(
            d_all_row_ptr,   // const int64_t* __restrict__ row_ptr, // CSR
            d_all_col_ids,   // const int64_t* __restrict__ col_ids, // CSR | COO
            d_all_row_ids,   // const int64_t* __restrict__ row_ids, // COO
            d_generators,    // const monom_t* __restrict__ generators,
            d_level_offsets, // const int64_t* __restrict__ level_offsets,
            masked_vertices, // const bool* __restrict__ masked_vertices,
            level,           // int level
            num_nodes,       // int num_nodes,
            num_edges,       // int64_t num_edges
            nullptr,         // int* count,
            irr_edges_cu     // int* irreducible_edges
        );

        vector<int> irr_edges(2*num_edges);
        cudaMemcpy(irr_edges.data(), irr_edges_cu, 2*num_edges*sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(irr_edges_cu);

        for (int64_t i = 0; i < num_edges; ++i) {
            if (irr_edges[2*i] == -1) continue;
            irreducible.push_back(irr_edges[2*i]);
            irreducible.push_back(irr_edges[2*i+1]);
        }

        #ifdef SGRAPH_DEBUG_
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error: %s. Attempted to pass: |E|=%lld, |V|=%d\n", cudaGetErrorString(err), num_edges, min(num_nodes, 1024));
            }
        #endif
    }

    #ifdef SGRAPH_DEBUG_
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    #endif
}


}  // syzygy
