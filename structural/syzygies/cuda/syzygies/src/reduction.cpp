#include <Python.h>
#include <ATen/Operators.h>

#ifdef BUILD_SGRAPH_CU_
    #include <c10/cuda/CUDAGuard.h>
#endif  // BUILD_SGRAPH_CU_

#include <torch/all.h>
#include <torch/library.h>
#include <torch/custom_class.h>

#include <deque>
#include <vector>
#include <memory>
#include <iostream>

#include "sgraph.hpp"

extern "C" {
    PyObject* PyInit__C(void) {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL
        };
        return PyModule_Create(&module_def);
    }
}

class SGraph_w : public torch::CustomClassHolder, public syzygy::SGraph {
 public:
    SGraph_w(int64_t d, int64_t n, int64_t max_level) : syzygy::SGraph(d, n, max_level) {}

    int64_t count_irreducible_cpu(at::Tensor mask) {
        TORCH_CHECK(mask.dtype() == at::kBool);
        at::Tensor mask_contg = mask.contiguous();

        return this->count_irreducible(mask_contg.data_ptr<bool>());
    }
};

at::Tensor create_sgraph_cpu(at::Tensor n, at::Tensor d, int64_t max_level) {
    TORCH_INTERNAL_ASSERT(n.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(d.device().type() == at::DeviceType::CPU);
    TORCH_CHECK(n.numel() == 1);
    TORCH_CHECK(d.numel() == 1);
    return at::tensor(reinterpret_cast<int64_t>(new syzygy::SGraph(d.item<int64_t>(), n.item<int64_t>(), max_level)));
}

int64_t count_irreducible_cpu(at::Tensor mask, int64_t sgraph_cache) {
    TORCH_CHECK(mask.dtype() == at::kBool);
    syzygy::SGraph* sgraph_ptr = reinterpret_cast<syzygy::SGraph*>(sgraph_cache);

    at::Tensor mask_contg = mask.contiguous();
    return sgraph_ptr->count_irreducible(mask_contg.data_ptr<bool>());
}

at::Tensor find_irreducible_cpu(at::Tensor mask, int64_t sgraph_cache) {
    TORCH_CHECK(mask.dtype() == at::kBool);
    syzygy::SGraph* sgraph_ptr = reinterpret_cast<syzygy::SGraph*>(sgraph_cache);

    at::Tensor mask_contg = mask.contiguous();

    vector<int> irreducible;
    sgraph_ptr->irreducible_edges(mask_contg.data_ptr<bool>(), irreducible);

    const int64_t num_edges = irreducible.size() / 2;

    return at::from_blob(
        irreducible.data(),
        {num_edges, 2},
        at::TensorOptions().dtype(at::kInt)).clone(); // TODO is int64_t worth it?
}

int64_t free_sgraph_cpu(at::Tensor sgraph_cache) {
    TORCH_CHECK(sgraph_cache.numel() == 1);
    delete reinterpret_cast<syzygy::SGraph*>(sgraph_cache.item<int64_t>());
    return 0;
}

int64_t count_irreducible_recompute_cpu(at::Tensor mask, int64_t n, int64_t d, int64_t max_level) {
    TORCH_CHECK(mask.dtype() == at::kBool);

    syzygy::SGraph sgraph(d, n, max_level);
    at::Tensor mask_contg = mask.contiguous();
    return sgraph.count_irreducible(mask_contg.data_ptr<bool>());
}

int64_t num_generators(at::Tensor sgraph_cache) {
    syzygy::SGraph* sgraph_ptr = reinterpret_cast<syzygy::SGraph*>(sgraph_cache.item<int64_t>());
    return sgraph_ptr->num_generators();
}

TORCH_LIBRARY(syzygies, m) {
    m.def("create_sgraph(Tensor n, Tensor d, int max_level) -> Tensor");
    m.def("count_irreducible(Tensor mask, int sgraph_cache) -> int");
    m.def("find_irreducible(Tensor mask, int sgraph_cache) -> Tensor");
    m.def("free_sgraph(Tensor sgraph_cache) -> int");
    m.def("num_generators(Tensor sgraph_cache) -> int");

    m.def("count_irreducible_recompute(Tensor mask, int n, int d, int max_level) -> int");

    m.class_<SGraph_w>("SGraph")
    .def(torch::init<int64_t, int64_t, int64_t>())
    .def("count_irreducible", &SGraph_w::count_irreducible_cpu);
}

TORCH_LIBRARY_IMPL(syzygies, CPU, m) {
    m.impl("create_sgraph", &create_sgraph_cpu);
    m.impl("count_irreducible", &count_irreducible_cpu);
    m.impl("find_irreducible", &find_irreducible_cpu);
    m.impl("free_sgraph", &free_sgraph_cpu);
    m.impl("num_generators", &num_generators);

    m.impl("count_irreducible_recompute", &count_irreducible_recompute_cpu);
}


#ifdef BUILD_SGRAPH_CU_

at::Tensor create_sgraph_gpu(at::Tensor n, at::Tensor d, int64_t max_level) {
    TORCH_INTERNAL_ASSERT(n.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(d.device().type() == at::DeviceType::CUDA);

    // Needed when there are multiple gpus
    c10::cuda::CUDAGuard device_guard(n.device());

    syzygy::SGraph* sgraph = new syzygy::SGraph(d.to(at::kCPU).item<int64_t>(), n.to(at::kCPU).item<int64_t>(), max_level);
    sgraph->fill_buffers_cu();
    return at::tensor(reinterpret_cast<int64_t>(sgraph));
}

int64_t count_irreducible_gpu(at::Tensor mask, int64_t sgraph_cache) {
    TORCH_CHECK(mask.dtype() == at::kBool);
    TORCH_INTERNAL_ASSERT(mask.device().type() == at::DeviceType::CUDA);
    syzygy::SGraph* sgraph_ptr = reinterpret_cast<syzygy::SGraph*>(sgraph_cache);

    at::Tensor mask_contg = mask.contiguous();
    return sgraph_ptr->count_irreducible_cu(mask_contg.data_ptr<bool>());
}

at::Tensor find_irreducible_gpu(at::Tensor mask, int64_t sgraph_cache) {
    TORCH_CHECK(mask.dtype() == at::kBool);
    TORCH_INTERNAL_ASSERT(mask.device().type() == at::DeviceType::CUDA);
    syzygy::SGraph* sgraph_ptr = reinterpret_cast<syzygy::SGraph*>(sgraph_cache);

    at::Tensor mask_contg = mask.contiguous();

    vector<int> irreducible;
    sgraph_ptr->irreducible_edges_cu(mask_contg.data_ptr<bool>(), irreducible);

    const int64_t num_edges = irreducible.size() / 2;

    return at::from_blob(
        irreducible.data(),
        {num_edges, 2},
        at::TensorOptions().dtype(at::kInt)).clone(); // TODO is int64_t worth it?
}

TORCH_LIBRARY_IMPL(syzygies, CUDA, m) {
    m.impl("create_sgraph", &create_sgraph_gpu);
    m.impl("count_irreducible", &count_irreducible_gpu);
    m.impl("find_irreducible", &find_irreducible_gpu);
}

#endif  // BUILD_SGRAPH_CU_
