
#ifndef SORTD_SOLUTION_TRACKER_H
#define SORTD_SOLUTION_TRACKER_H

#include "base.h"
#include "model/data.h"
#include "solver/tree.h"

namespace SORTD {

    template <class OT>
    class Solver;

    template<class OT, class SVT>
    struct AbstractSolutionTracker;

    template <typename OT>
    using SolutionTrackerP = std::shared_ptr<AbstractSolutionTracker<OT>>;
    
    // Abstract Solution Tracker for the optimization task OT with solution value type SVT (by default OT::SolType)
    template <class OT, class SVT>
    struct AbstractSolutionTracker {
        using SolType = typename OT::SolType;
        using SolLabelType = typename OT::SolLabelType;

        AbstractSolutionTracker(SVT obj) : obj(obj) {}

        virtual bool IsLeaf() const { return false; }
        virtual bool IsD1() const { return false; }
        virtual bool IsRecursive() const { return false; }
        virtual SolLabelType GetLabel() const {return OT::worst_label;}
        virtual SolLabelType GetAltLabel() const {return OT::worst_label;}
        virtual void SetAltLabel(SolLabelType _alt_label) {return;}
        virtual void SwitchLabel() {return;}
        virtual size_t GetNumSolutions() const = 0;
        virtual int GetFeature() const { return -1; }
        virtual size_t GetNodeCount(int node_budget) const = 0;
        virtual size_t GetFeatureCount(int feature) const = 0;
        virtual std::vector<size_t> GetNodeCounts() const = 0;
        virtual std::vector<size_t> GetCumulativeSolCount() const { return {};};
        virtual std::vector <std::pair<std::shared_ptr<AbstractSolutionTracker<OT, SVT>>, std::shared_ptr<AbstractSolutionTracker<OT, SVT>>>> GetSolutions() const { return {};};

        virtual void UpdateNodeCount(int node_budget, size_t amount) {}

        virtual void CalculateFeatureStats() {}
        virtual void CalculateNodeStats() {}

        virtual std::shared_ptr<Tree<OT>> CreateTreeWithIndexN(const Solver<OT>* solver, size_t n) const = 0;

        virtual bool HasQueryFeatureF(int f) const { return false; }
        virtual size_t NumberOfQueryFeatureF(int f) const { return 0; }
        virtual std::vector<std::shared_ptr<Tree<OT>>> CreateQueryTreesWithFeature(const Solver<OT>* solver, int query_feature) { return {}; }
        virtual std::vector<std::shared_ptr<Tree<OT>>> CreateQueryTreesWithoutFeature(const Solver<OT>* solver, int query_feature) { return { CreateTreeWithIndexN(solver, 0) }; }
        virtual std::vector<std::shared_ptr<Tree<OT>>> CreateNodeBudgetQueryTrees(const Solver<OT>* solver, int node_budget) = 0;

        virtual void Shrink() {}

        SVT obj;
    };

    // LeafSolutionTracker stores leaf solutions
    template <class OT, class SVT = typename OT::SolType>
    struct LeafSolutionTracker : public AbstractSolutionTracker<OT, SVT> {
        using SolType = typename OT::SolType;
        using SolLabelType = typename OT::SolLabelType;

        LeafSolutionTracker(SVT obj, const SolLabelType& label) : AbstractSolutionTracker<OT, SVT>(obj), label(label) {}
        
        inline bool IsLeaf() const override { return true; }
        inline size_t GetNumSolutions() const override { return 1; }
        inline size_t GetNodeCount(int node_budget) const override { return node_budget == 0 ? 1 : 0; }
        inline size_t GetFeatureCount(int feature) const override { return 0; }
        inline std::vector<size_t> GetNodeCounts() const override { return { 1 }; }
        inline SolLabelType GetLabel() const override { return label; }
        inline SolLabelType GetAltLabel() const override {return alt_label;}
        void SetAltLabel(SolLabelType _alt_label) override {
            alt_label = _alt_label;
        }

        void SwitchLabel() override {
            runtime_assert(label != alt_label);
            std::swap(label, alt_label);
        }

        std::shared_ptr<Tree<OT>> CreateTreeWithIndexN(const Solver<OT>* solver, size_t n) const override {
            runtime_assert(n == 0);
            return Tree<OT>::CreateLabelNode(label);
        }

        std::vector<std::shared_ptr<Tree<OT>>> CreateNodeBudgetQueryTrees(const Solver<OT>* solver, int node_budget) override {
            if (node_budget == 0) return { CreateTreeWithIndexN(solver, 0) };
            return {};
        }

        SolLabelType label;
        SolLabelType alt_label = OT::worst_label;
    };

    // D1SolutionTracker stores solutions with a single branching node
    template <class OT, class SVT = typename OT::SolType>
    struct D1SolutionTracker : public AbstractSolutionTracker<OT, SVT> {
        using SolType = typename OT::SolType;
        using SolLabelType = typename OT::SolLabelType;

        D1SolutionTracker(SVT obj, int feature, const SolLabelType& left_label, const SolLabelType& right_label)
            : AbstractSolutionTracker<OT, SVT>(obj), feature(feature), left_label(left_label), right_label(right_label) {

//            runtime_assert(left_label != right_label);
        }

        inline bool IsD1() const override { return true; }
        inline int GetFeature() const override { return feature; }
        inline size_t GetNumSolutions() const override { return 1; }
        inline size_t GetNodeCount(int node_budget) const override{ return node_budget == 1 ? 1 : 0; }
        inline size_t GetFeatureCount(int feature) const override { return this->feature == feature ? 1 : 0; }
        inline bool HasQueryFeatureF(int f) const override { return feature == f; }
        inline std::vector<size_t> GetNodeCounts() const override { return { 0, 1 }; }

        std::shared_ptr<Tree<OT>> CreateTreeWithIndexN(const Solver<OT>* solver, size_t n) const override{
            runtime_assert(n == 0);
            auto tree = Tree<OT>::CreateFeatureNodeWithNullChildren(feature);
            tree->left_child = Tree<OT>::CreateLabelNode(left_label);
            tree->right_child = Tree<OT>::CreateLabelNode(right_label);
            if (solver->IsFeatureFlipped(feature)) {
                std::swap(tree->left_child, tree->right_child);
            }
            return tree;
        }

        inline size_t NumberOfQueryFeatureF(int f) const override { return feature == f ? 1 : 0; }
        
        std::vector<std::shared_ptr<Tree<OT>>>  CreateQueryTreesWithFeature(const Solver<OT>* solver, int query_feature) override {
            if (feature == query_feature) return { CreateTreeWithIndexN(solver, 0) };
            return {};
        }

        std::vector<std::shared_ptr<Tree<OT>>>  CreateQueryTreesWithoutFeature(const Solver<OT>* solver, int query_feature) override {
            if (feature != query_feature) return { CreateTreeWithIndexN(solver, 0) };
            return {};
        }

        std::vector<std::shared_ptr<Tree<OT>>> CreateNodeBudgetQueryTrees(const Solver<OT>* solver, int node_budget) override {
            if (node_budget == 1) return { CreateTreeWithIndexN(solver, 0) };
            return {};
        }


        int feature;
        SolLabelType left_label = OT::worst_label;
        SolLabelType right_label = OT::worst_label;
    };

    // RecursiveSolutionTracker stores solutions with more than one branching node
    template<class OT, class SVT = typename OT::SolType>
    struct RecursiveSolutionTracker : public AbstractSolutionTracker<OT, SVT> {
        using SolType = typename OT::SolType;
        using SolLabelType = typename OT::SolLabelType;

        RecursiveSolutionTracker(SVT obj, int feature, int num_features) : AbstractSolutionTracker<OT, SVT>(obj), feature(feature) {
            num_solutions = 0;
            feature_count.resize(num_features);
        }            

        bool IsRecursive() const override { return true; }
        inline size_t GetNumSolutions() const override { return num_solutions; }
        inline int GetFeature() const override { return feature; }
        inline size_t GetFeatureCount(int feature) const override { return feature_count[feature]; }
        inline std::vector<size_t> GetNodeCounts() const override { return num_nodes_count; }
        inline std::vector<size_t> GetCumulativeSolCount() const override {return cumulative_sol_count;}
        inline std::vector <std::pair<std::shared_ptr<AbstractSolutionTracker<OT, SVT>>, std::shared_ptr<AbstractSolutionTracker<OT, SVT>>>> GetSolutions() const override {return solutions;}

        /*
        * Shrink the vectors in this object to fit in the minimum required memory
        */
        void Shrink() override {
            solutions.shrink_to_fit();
            cumulative_sol_count.shrink_to_fit();
        }

        void CalculateFeatureStats() override;

        void CalculateNodeStats() override;

        size_t GetNodeCount(int node_budget) const override {
            if (node_budget >= num_nodes_count.size()) return 0;
            return num_nodes_count[node_budget];
        }

        void UpdateNodeCount(int node_budget, size_t amount) override {
            if (node_budget >= num_nodes_count.size()) num_nodes_count.resize(node_budget + 1, 0);
            num_nodes_count[node_budget] += amount;
        }

        void UpdateNumSolutions(size_t amount) { num_solutions += amount; }

        std::shared_ptr<Tree<OT>> CreateTreeWithIndexN(const Solver<OT>* solver, size_t n) const override;

        inline bool HasQueryFeatureF (int f) const override {
            return (feature_count.empty() && feature == f) || (!feature_count.empty() && feature_count[f] > 0);
        }

        size_t NumberOfQueryFeatureF(int f) const override;

        std::vector<std::shared_ptr<Tree<OT>>> CreateQueryTreesWithFeature(const Solver<OT>* solver, int query_feature) override;

        std::vector<std::shared_ptr<Tree<OT>>> CreateQueryTreesWithoutFeature(const Solver<OT>* solver, int query_feature) override;

        std::vector<std::shared_ptr<Tree<OT>>> CreateNodeBudgetQueryTrees(const Solver<OT>* solver, int node_budget) override;

        int feature;
        size_t num_solutions = 0;

        std::vector<size_t> feature_count;
        std::vector<size_t> num_nodes_count;
        std::vector<size_t> cumulative_sol_count;

        std::vector <std::pair<std::shared_ptr<AbstractSolutionTracker<OT, SVT>>, std::shared_ptr<AbstractSolutionTracker<OT, SVT>>>> solutions;
    };

    template <class OT>
    struct CompareSolutionTrackers {
        bool operator()(const SolutionTrackerP<OT> a, const SolutionTrackerP<OT> b) const {
            if (SOL_EQUAL(typename OT::SolType, a->obj, b->obj)) {
                return a->GetFeature() < b->GetFeature();
            }
            return a->obj < b->obj;
        }
    };

    template <class OT>
    struct SolutionTrackerEqual {

        bool operator()(const SolutionTrackerP<OT>& lhs, const SolutionTrackerP<OT>& rhs) const {
            if (!lhs || !rhs) return lhs == rhs;
            runtime_assert(!dynamic_cast<RecursiveSolutionTracker<OT>*>(lhs.get())); // We only want to compare and store solution trackers that don't recurse
            runtime_assert(!dynamic_cast<RecursiveSolutionTracker<OT>*>(rhs.get())); // We only want to compare and store solution trackers that don't recurse
            auto _lhs = dynamic_cast<LeafSolutionTracker<OT>*>(lhs.get());
            auto _rhs = dynamic_cast<LeafSolutionTracker<OT>*>(rhs.get());
            if (_lhs && _rhs) {
                if constexpr (std::is_same<typename OT::SolType, double>::value) {
                    return _lhs->label == _rhs->label
                        && _lhs->alt_label == _rhs->alt_label
                        && SOL_EQUAL(typename OT::SolType, _lhs->obj, _rhs->obj);
                }
                return _lhs->label == _rhs->label
                    && _lhs->alt_label == _rhs->alt_label
                    && _lhs->obj == _rhs->obj;
            } 
            auto __lhs = dynamic_cast<D1SolutionTracker<OT>*>(lhs.get());
            auto __rhs = dynamic_cast<D1SolutionTracker<OT>*>(rhs.get());
            if (__lhs && __rhs) {
                if constexpr (std::is_same<typename OT::SolType, double>::value) {
                    return __lhs->feature == __rhs->feature
                        && __lhs->left_label == __rhs->left_label
                        && __lhs->right_label == __rhs->right_label
                        && SOL_EQUAL(typename OT::SolType, __lhs->obj, __rhs->obj);
                }
                return __lhs->feature == __rhs->feature
                    && __lhs->left_label == __rhs->left_label
                    && __lhs->right_label == __rhs->right_label
                    && __lhs->obj == __rhs->obj;
            }
            return false;
        }
    };

    template <class OT>
    struct SolutionTrackerHash {

        std::size_t operator()(const SolutionTrackerP<OT>& solution) const {
            //adapted from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
            if (!solution) return 0;
            using std::size_t;
            using std::hash;
            // We only want to hash and store solution trackers that don't recurse
            runtime_assert(!dynamic_cast<RecursiveSolutionTracker<OT>*>(solution.get())); 
            size_t seed = hash<typename OT::SolType>()(solution->obj);
            if (auto _solution = dynamic_cast<LeafSolutionTracker<OT>*>(solution.get())) {
                seed ^= hash<typename OT::SolLabelType>()(_solution->label) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            } else if (auto _solution = dynamic_cast<D1SolutionTracker<OT>*>(solution.get())) {
                seed ^= hash<int>()(_solution->feature) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                seed ^= hash<typename OT::SolLabelType>()(_solution->left_label) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                seed ^= hash<typename OT::SolLabelType>()(_solution->right_label) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;

        }
    };
}

#endif //SORTD_SOLUTION_TRACKER_H
