/**
 * @file sgraph.cpp
 * @brief Implements SGraph.hpp
 */

 #include "sgraph.hpp"


namespace syzygy {

// PRIVATE

int64_t choose(int64_t n, int64_t k) { return (k == 0) ? 1 : (n * choose(n - 1, k - 1) / k); }

monom_t vec_to_gen(vector<bool>& generator) {
    monom_t gen_bin = 0;
    for (int i = 0; i < generator.size(); ++i) gen_bin |= generator[i]*(1<<i);
    return gen_bin;
}

int64_t SGraph::num_edges(int level) {
    /**
     * @brief Easy to prove. For proof see prop 5.1 in
     *  https://sas.uwaterloo.ca/~rwoldfor/papers/navigation/cliques/johnson_graphs/johnson_cliquesDRAFT.pdf
     */
    int64_t k = d - level;
    return choose(n, k) * choose(n - k, d - k) * choose(n - d, d - k) / 2;
}

int64_t SGraph::num_generators() {
    return this->generators.size();
}

void SGraph::load_generators(vector<monom_t>& generators) {
    vector<bool> generator(this->n);
    std::fill(generator.begin(), generator.begin() + this->d, true);
    generators.push_back(vec_to_gen(generator));

    while (std::prev_permutation(generator.begin(), generator.end())) generators.push_back(vec_to_gen(generator));
}

void SGraph::build_graph(vector<monom_t>& generators, int64_t** col_ids, int64_t** row_ptr, int max_level) {
    int N = generators.size();

    for (int level = 1; level <= max_level; ++level) row_ptr[level-1][0] = 0;

    for (int64_t i = 0; i < N; ++i) {
        int* edge_count = new int[max_level];
        for (int level = 1; level <= max_level; ++level) edge_count[level-1] = 0;
        for (int64_t j = 0; j < N; ++j) {
            // TODO: digraph or graph for CSR? direction is redundant here, might be optimized further
            if (i == j) continue;
            monom_t e_i = generators[j] & (~generators[i]);
            int level = __builtin_popcount(e_i);
            if (level > max_level) continue;

            col_ids[level-1][row_ptr[level-1][i] + edge_count[level-1]++] = j;
        }
        for (int level = 1; level <= max_level; ++level)
            row_ptr[level-1][i+1] = row_ptr[level-1][i] + edge_count[level - 1];

        // free(edge_count);
        delete [] edge_count;
    }
}

// PUBLIC

SGraph::SGraph(int d, int n, int max_level) : n(n), d(d), max_level(max_level) {
    if (n > 2*d && max_level < d) std::cerr << "Max level is too small. Highest level generators might be irreducible, but will be ignored\n";
    load_generators(this->generators);
    std::sort(this->generators.begin(), this->generators.end());

    h_all_row_ptr = reinterpret_cast<int64_t**>(malloc(sizeof(int64_t*)*max_level));
    h_all_col_ids = reinterpret_cast<int64_t**>(malloc(sizeof(int64_t*)*max_level));

    for (int level = 1; level <= max_level; ++level) {
        h_all_row_ptr[level-1] = reinterpret_cast<int64_t*>(malloc(sizeof(int64_t)*(generators.size()+1)));
        h_all_col_ids[level-1] = reinterpret_cast<int64_t*>(malloc(sizeof(int64_t)*2*this->num_edges(level)));
    }

    build_graph(this->generators, h_all_col_ids, h_all_row_ptr, max_level);
}

SGraph::~SGraph() {
    for (int level = 1; level <= this->max_level; ++level) {
        free(h_all_row_ptr[level-1]);
        free(h_all_col_ids[level-1]);
    }
    free(h_all_row_ptr);
    free(h_all_col_ids);

    cuda_deallocate();
}

monom_t SGraph::weight(int64_t s_idx, int64_t t_idx) {
    return generators[t_idx] & (~generators[s_idx]);
}

bool SGraph::is_reducible(int64_t row_idx, int64_t col_idx, int64_t* row_ptr, int64_t* col_ids, bool* masked_vertices) {
    // assert(row_idx < generators.size());
    // assert(col_idx < generators.size());
    std::queue<std::pair<monom_t, int64_t>> queue;

    bool* visited = new bool[this->generators.size()];
    for (int i = 0; i < this->generators.size(); ++i) visited[i] = false;

    queue.push(std::make_pair(row_idx, weight(row_idx, col_idx)));

    while (!queue.empty()) {
        auto& [s, s_gen] = queue.front();
        queue.pop();

        for (int k_idx = row_ptr[s]; k_idx < row_ptr[s+1]; ++k_idx) {
            int64_t k = col_ids[k_idx];
            // assert(k < generators.size());
            if (visited[k] || !masked_vertices[k]) continue;

            if ((s_gen | weight(s, k)) == s_gen) {
                monom_t prod = s_gen & (~weight(s, k));
                if (k == col_idx) return true;

                queue.push(std::make_pair(k, weight(k, s) | prod));
            }
        }
        visited[s] = true;
    }

    // free(visited);
    delete [] visited;

    return false;
}

int64_t SGraph::count_irreducible(bool* masked_vertices) {
    int64_t count = 0;

    for (int level = 2; level <= this->max_level; ++level) {
        for (int i = 0; i < generators.size(); ++i) {
            if (!masked_vertices[i]) continue;

            for (int j = h_all_row_ptr[level-1][i]; j < h_all_row_ptr[level-1][i+1]; ++j) {
                if (i > h_all_col_ids[level-1][j]) continue;
                if (!masked_vertices[h_all_col_ids[level-1][j]]) continue;

                if (!is_reducible(i, h_all_col_ids[level-1][j], h_all_row_ptr[0], h_all_col_ids[0], masked_vertices)) ++count;
            }
        }
    }

    return count;
}

void SGraph::irreducible_edges(bool* masked_vertices, vector<int>& irreducible) {
    for (int level = 2; level <= this->max_level; ++level) {
        for (int i = 0; i < generators.size(); ++i) {
            if (!masked_vertices[i]) continue;

            for (int j = h_all_row_ptr[level-1][i]; j < h_all_row_ptr[level-1][i+1]; ++j) {
                if (i > h_all_col_ids[level-1][j]) continue;
                if (!masked_vertices[h_all_col_ids[level-1][j]]) continue;

                if (!is_reducible(i, h_all_col_ids[level-1][j], h_all_row_ptr[0], h_all_col_ids[0], masked_vertices)) {
                    irreducible.push_back(i); irreducible.push_back(h_all_col_ids[level-1][j]);
                }
            }
        }
    }
}


// weak
void __attribute__((weak)) SGraph::cuda_deallocate() {}
void __attribute__((weak)) SGraph::fill_buffers_cu() {}
int64_t __attribute__((weak)) SGraph::count_irreducible_cu(bool* masked_vertices) { return 0; }

}  // syzygy
