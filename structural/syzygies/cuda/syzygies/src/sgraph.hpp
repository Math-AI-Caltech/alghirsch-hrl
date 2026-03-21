#ifndef SGRAPH_HPP_
#define SGRAPH_HPP_

#include <queue>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdint>
using std::vector;


namespace syzygy {


typedef int monom_t;


class SGraph {
 private:
    int n, d;
    int max_level;

    vector<monom_t> generators;

    int64_t num_edges(int level);

    void load_generators(vector<monom_t>& generators);

    void build_graph(vector<monom_t>& generators, int64_t** col_ids, int64_t** row_ptr, int max_level);

    void cuda_deallocate();

    int64_t** h_all_row_ptr;
    int64_t** h_all_col_ids;

    bool cu_buffers_filled = false;
    int64_t* d_all_row_ptr;
    int64_t* d_all_col_ids;
    monom_t* d_generators;

    // Useful for parallizing over edges: COO (access) + CSR (BFS).
    int64_t* d_all_row_ids;
    int64_t* d_level_offsets;
    int* d_count;

 public:
    SGraph(int d, int n, int max_level); // default: max_level = 2

    ~SGraph();

    void fill_buffers_cu();

    monom_t weight(int64_t s_idx, int64_t t_idx);

    bool is_reducible(int64_t row_idx, int64_t col_idx, int64_t* row_ptr, int64_t* col_ids, bool* masked_vertices);

    /**
     * @brief Counts the number of irreducible edges at all levels 1 < level ≤ max_level.
     *
     * @param masked_vertices Assumes points to a memory of size = generators.size();
     * @return int64_t Number of irreducible edges at levels > 1 and ≤ max_level.
     */
    int64_t count_irreducible(bool* masked_vertices);

    /**
     * @brief Runs reduction algorithm for each max_level ≥ level ≥ 2 edge.
     *
     * @param masked_vertices Assumes points to a memory of size = generators.size();
     */
    void irreducible_edges(bool* masked_vertices, vector<int>& irreducible);

    int64_t num_generators();

   //////////////////////////
   /////// BEGIN CUDA ///////
   //////////////////////////

    /**
     * @brief Counts the number of irreducible edges at all levels 1 < level ≤ max_level. (CUDA version).
     *
     * @param masked_vertices Assumes points to device.
     */
    int64_t count_irreducible_cu(bool* masked_vertices);

    /**
     * @brief Runs reduction algorithm for each max_level ≥ level ≥ 2 edge. (CUDA version)
     *
     * @param masked_vertices Assumes points to a memory of size = generators.size();
     */
    void irreducible_edges_cu(bool* masked_vertices, vector<int>& irreducible);

    ////////////////////////
    /////// END CUDA ///////
    ////////////////////////
};


}  // syzygy

#endif  // SGRAPH_HPP_
