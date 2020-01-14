// Functions for constructing and working with the MST coordinate system

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <mpi.h>


#include "ketju_integrator/ketju_integrator.h"
#include "ketju_integrator/internal.h"

int ketju_check_relative_proximity_ND_2(int v1, int v2,
                                        const struct ketju_system *R, int *d,
                                        int *path, int *sign) {

    const struct spanning_tree_coordinate_system *mst = R->internal_state->mst;

    int diff = abs(mst->vertex[v1].level - mst->vertex[v2].level);

    if (diff == 1) {
        if (mst->vertex[v2].parent == v1) {
            *d = 1;
            sign[0] = -1;
            path[0] = mst->vertex[v2].edgetoparent;
            return 1;
        } else if (mst->vertex[v1].parent == v2) {
            *d = 1;
            sign[0] = +1;
            path[0] = mst->vertex[v1].edgetoparent;
            return 1;
        }
        return 0;
    } else if (diff == 2) {
        if (mst->vertex[mst->vertex[v2].parent].parent == v1) {
            *d = 2;
            sign[0] = -1;
            sign[1] = -1;
            path[0] = mst->vertex[v2].edgetoparent;
            path[1] = mst->vertex[mst->vertex[v2].parent].edgetoparent;
            return 1;
        } else if (mst->vertex[mst->vertex[v1].parent].parent == v2) {
            *d = 2;
            sign[0] = +1;
            sign[1] = +1;
            path[0] = mst->vertex[v1].edgetoparent;
            path[1] = mst->vertex[mst->vertex[v1].parent].edgetoparent;
            return 1;
        }
        return 0;
    } else if (diff == 0) {
        if (mst->vertex[v1].parent == mst->vertex[v2].parent) {
            *d = 2;
            sign[0] = -1;
            sign[1] = +1;
            path[0] = mst->vertex[v2].edgetoparent;
            path[1] = mst->vertex[v1].edgetoparent;
            return 1;
        }
        return 0;
    }
    return 0;
}

int ketju_check_relative_proximity(int v1, int v2, const int Nd,
                             const struct ketju_system *R, int *d, int *path,
                             int *sign) {

    const struct spanning_tree_coordinate_system *mst = R->internal_state->mst;

    for (int i = 0; i < Nd; i++) {
        path[i] = -666;
        sign[i] = 0;
    }

    int proximity = 0;
    if (mst->vertex[v1].level == mst->vertex[v2].level) {

        if (mst->vertex[v1].level == 0) {
            char buf[100];
            sprintf(buf,
                    "ketju_check_relative_proximity: This should not happen! "
                    "thistask: %d --- v1 v2 %d %d --- levels: %d %d\n",
                    R->internal_state->task_info->global_rank, mst->vertex[v1].id, mst->vertex[v2].id,
                    mst->vertex[v1].level, mst->vertex[v2].level);
            terminate(buf);
        }

        int halfNd = (int)floor(0.5 * Nd);
        int *pathend = calloc(Nd, sizeof(int));

        for (int i = 0; i < halfNd; i++) {
            sign[i] = +1;
            path[i] = mst->vertex[v1].edgetoparent;
            pathend[i] = mst->vertex[v2].edgetoparent;
            v1 = mst->vertex[v1].parent;
            v2 = mst->vertex[v2].parent;
            if (v1 == v2) {
                *d = 2 * (i + 1);
                int c = 0;
                for (int m = i + 1; m < (*d); m++) {
                    sign[m] = -1;
                    path[m] = pathend[i - c];
                    c++;
                }
                proximity = 1;
                break;
            }
        }

        free(pathend);

    } else {
        *d = abs(mst->vertex[v1].level - mst->vertex[v2].level);
        if (*d <= Nd) {
            int lo, hi, thissign;
            if (mst->vertex[v1].level > mst->vertex[v2].level) {
                lo = v2;
                hi = v1;
                thissign = +1;
            } else {
                lo = v1;
                hi = v2;
                thissign = -1;
            }
            for (int i = 0; i < *d; i++) {
                path[i] = mst->vertex[hi].edgetoparent;
                sign[i] = thissign;
                hi = mst->vertex[hi].parent;
            }
            if (lo == hi) { proximity = 1; }
        }
    }

    return proximity;
}


#if 0
void get_v1_v2(int N, int target, int *v1, int *v2) {

    int final_element_on_row = 0;
    int elements_on_this_row;

    for (int row = 0; row < N - 1; row++) {
        if (row == 0) {
            elements_on_this_row = N - 2 - row;
        } else {
            elements_on_this_row = N - 1 - row;
        }
        final_element_on_row += elements_on_this_row;
        if (target <= final_element_on_row) {
            *v1 = row;
            int diff = final_element_on_row - target;
            *v2 = N - 1 - diff;
            break;
        }
    }
}

int cmp_weight_index2(const void *a, const void *b) {
    struct WeightIndex2 *a1 = (struct WeightIndex2 *)a;
    struct WeightIndex2 *a2 = (struct WeightIndex2 *)b;
    if ((*a1).weight > (*a2).weight)
        return 1;
    else if ((*a1).weight < (*a2).weight)
        return -1;
    else
        return 0;
}

int cmp(const void *a, const void *b) {
    struct GraphEdge *a1 = (struct GraphEdge *)a;
    struct GraphEdge *a2 = (struct GraphEdge *)b;
    if ((*a1).weight > (*a2).weight)
        return 1;
    else if ((*a1).weight < (*a2).weight)
        return -1;
    else
        return 0;
}

int EdgeSearchForPrim(struct ketju_system *R, int NumLocalEdges) {

    clock_t edgesearch_time0 = clock();
    struct WeightIndex wi;

    wi.weight = 1e10;
    wi.index = -1;
    for (int i = 0; i < NumLocalEdges; i++) {
        if (R->Vertex[R->LocalEdge[i].vertex1].inMST !=
            R->Vertex[R->LocalEdge[i].vertex2].inMST) {
            wi.weight = R->LocalEdge[i].weight;
            wi.index = R->LocalEdge[i].id;
            break;
        }
    }

    clock_t edgesearch_time1 = clock();
    tprim3 += (double)(edgesearch_time1 - edgesearch_time0) / CLOCKS_PER_SEC;

    edgesearch_time0 = clock();
#ifdef PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, &wi, 1, MPI_DOUBLE_INT, MPI_MINLOC,
                  MPI_COMM_WORLD);
#endif

    edgesearch_time0 = clock();
    tprim4 += (double)(edgesearch_time1 - edgesearch_time0) / CLOCKS_PER_SEC;

    return wi.index;
}

void PrimMST(struct ketju_system *R) {

    msttime0 = clock();

    int CumNumEdge = 0, istart = 0;
#ifdef PARALLEL
    int loop = 1, ThisBlock, jstart;
    loop_scheduling_N2(R->NumVertex, Ntask, &istart, &jstart, &ThisBlock,
                       &CumNumEdge, ThisTask);
#endif
    int NumLocalEdges = 0;
    int c = 0, counter = 0;

    double rmin = 1e10, r2;
    int last = -1;

    for (int i = 0; i < R->NumVertex; i++) {
        r2 = 0;
        for (int k = 0; k < 3; k++) {
            double ds = R->Pos[3 * i + k];
            r2 += ds * ds;
        }
        if (rmin > sqrt(r2)) {
            rmin = sqrt(r2);
            last = i;
        }
    }

    for (int i = istart; i < R->NumVertex; i++) {

#ifdef PARALLEL
        if (i > istart) { jstart = i + 1; }
        for (int j = jstart; j < R->NumVertex; j++) {
#else
        for (int j = i + 1; j < R->NumVertex; j++) {
#endif
            counter++;
            double w = 0;
            for (int k = 0; k < 3; k++) {
                double ds = R->Pos[3 * i + k] - R->Pos[3 * j + k];
                w += ds * ds;
            }
            w = sqrt(w);

            R->LocalEdge[NumLocalEdges].weight = w;
            R->LocalEdge[NumLocalEdges].vertex1 = i;
            R->LocalEdge[NumLocalEdges].vertex2 = j;
            R->LocalEdge[NumLocalEdges].id = (int)(CumNumEdge + c);

            NumLocalEdges++;
            c++;

#ifdef PARALLEL
            if (counter == ThisBlock) {
                loop = 0;
                break;
            }
        }
        if (loop == 0) { break; }
#else
        }
#endif
    }

    qsort(R->LocalEdge, NumLocalEdges, sizeof(struct GraphEdge), cmp);

    int remaining = 0;

    for (int i = 0; i < R->NumVertex; i++) {
        R->Vertex[i].id = i;
        R->Vertex[i].degree = 0;
        R->Vertex[i].edgetoparent = -1;
        R->Vertex[i].parent = -1;
        R->Vertex[i].level = 0;
        R->Vertex[i].inMST = 0;
    }

    R->Vertex[last].inMST = 1;

    c = 0;
    do {

        int NextEdge = EdgeSearchForPrim(R, NumLocalEdges);

        int v1 = -1, v2 = -1;
        get_v1_v2(R->NumVertex, NextEdge, &v1, &v2);

        R->EdgeInMST[c].vertex1 = v1;
        R->EdgeInMST[c].vertex2 = v2;

        R->Vertex[v1].degree++;
        R->Vertex[v2].degree++;

        if (R->Vertex[v1].inMST == 0) {

            R->Vertex[v1].parent = v2;
            R->Vertex[v1].level = R->Vertex[v2].level + 1;
            R->Vertex[v1].inMST = 1;
            R->Vertex[v1].edgetoparent = c;
            for (int k = 0; k < 3; k++) {
                R->State[3 * c + k] = R->Pos[3 * v1 + k] - R->Pos[3 * v2 + k];
                R->State[3 * (R->NumVertex - 1) + 3 * c + k] =
                    R->Vel[3 * v1 + k] - R->Vel[3 * v2 + k];
            }
        } else {

            R->Vertex[v2].parent = v1;
            R->Vertex[v2].level = R->Vertex[v1].level + 1;
            R->Vertex[v2].inMST = 1;
            R->Vertex[v2].edgetoparent = c;
            for (int k = 0; k < 3; k++) {
                R->State[3 * c + k] = R->Pos[3 * v2 + k] - R->Pos[3 * v1 + k];
                R->State[3 * (R->NumVertex - 1) + 3 * c + k] =
                    R->Vel[3 * v2 + k] - R->Vel[3 * v1 + k];
            }
        }
        c++;
        remaining++;
    } while (remaining < R->NumVertex - 1);

    msttime1 = clock();
    time_mst += (double)(msttime1 - msttime0) / CLOCKS_PER_SEC;
}

double get_rmax(struct ketju_system *R) {
    double rmax2 = 0;
    for (int i = 0; i < R->NumVertex; i++) {
        double ds;
        double r2 = 0;
        for (int k = 0; k < 3; k++) {
            ds = R->Pos[3 * i + k];
            r2 += ds * ds;
        }
        if (r2 > rmax2) { rmax2 = r2; }
    }
    return sqrt(rmax2);
}

void get_triplet(int index, int *triplet) {
    if (index >= 100) {
        triplet[2] = 1;
        index -= 100;
        if (index >= 10) {
            triplet[1] = 1;
            index -= 10;
            if (index == 1) {
                triplet[0] = 1;
            } else {
                triplet[0] = -1;
            }
        } else {
            triplet[1] = 0;
            if (index == 1) {
                triplet[0] = 1;
            } else {
                triplet[0] = -1;
            }
        }
    } else {
        triplet[2] = 0;
        if (index >= 10) {
            triplet[1] = 1;
            index -= 10;
            if (index == 1) {
                triplet[0] = 1;
            } else {
                triplet[0] = -1;
            }
        } else {
            triplet[1] = 0;
            if (index == 1) {
                triplet[0] = 1;
            } else {
                triplet[0] = -1;
            }
        }
    }
}

int get_triplet_index(int num) {
    if (num == 1) return 1;
    return 0;
}

int get_octant(int triplet[3]) {
    if (triplet[0] == 1) {
        if (triplet[1] == 1) {
            if (triplet[2] == -1) {
                return 6;
            } else {
                return 7;
            }
        } else {
            if (triplet[2] == -1) {
                return 4;
            } else {
                return 5;
            }
        }
    } else {
        if (triplet[1] == 1) {
            if (triplet[2] == -1) {
                return 2;
            } else {
                return 3;
            }
        } else {
            if (triplet[2] == -1) {
                return 0;
            } else {
                return 1;
            }
        }
    }
    return -1;
}

void get_triplet_from_octant(int octant, int *triplet) {

    if (octant == 0) {
        triplet[0] = -1;
        triplet[1] = -1;
        triplet[2] = -1;
        return;
    }
    if (octant == 1) {
        triplet[0] = -1;
        triplet[1] = -1;
        triplet[2] = +1;
        return;
    }
    if (octant == 2) {
        triplet[0] = -1;
        triplet[1] = +1;
        triplet[2] = -1;
        return;
    }
    if (octant == 3) {
        triplet[0] = -1;
        triplet[1] = +1;
        triplet[2] = +1;
        return;
    }
    if (octant == 4) {
        triplet[0] = +1;
        triplet[1] = -1;
        triplet[2] = -1;
        return;
    }
    if (octant == 5) {
        triplet[0] = +1;
        triplet[1] = -1;
        triplet[2] = +1;
        return;
    }
    if (octant == 6) {
        triplet[0] = +1;
        triplet[1] = +1;
        triplet[2] = -1;
        return;
    }
    if (octant == 7) {
        triplet[0] = +1;
        triplet[1] = +1;
        triplet[2] = +1;
        return;
    }
    return;
}

void list_octree_level(struct OctreeNode *Node, int *LeafSizes,
                       int **LeafParticles, int *Nleafs) {

    if (Node->IsLeaf) {
        LeafSizes[*Nleafs] = Node->Npart;
        LeafParticles[*Nleafs] = calloc(Node->Npart, sizeof(int));
        for (int i = 0; i < Node->Npart; i++) {
            LeafParticles[*Nleafs][i] = Node->ObjectList[i];
        }
        (*Nleafs)++;
    } else {
        for (int oct = 0; oct < 8; oct++) {
            if (Node->Children[oct]->ChildExists) {
                list_octree_level(Node->Children[oct], LeafSizes, LeafParticles,
                                  Nleafs);
            }
        }
    }
}

void GetNumberOfLeafs(struct OctreeNode *Node, int *Nleaf) {
    if (Node->IsLeaf) {
        (*Nleaf)++;
    } else {
        for (int oct = 0; oct < 8; oct++) {
            if (Node->Children[oct]->ChildExists) {
                GetNumberOfLeafs(Node->Children[oct], Nleaf);
            }
        }
    }
}

void free_octree(struct OctreeNode *Node) {
    for (int oct = 0; oct < 8; oct++) {
        if (Node->Children[oct]->ChildExists) {
            free_octree(Node->Children[oct]);
        }
    }
    free(Node->ObjectList);
    free(Node);
}

void allocate_new_octree_node(struct OctreeNode *Node, int oct, int *label) {

    Node->ChildExists[oct] = 1;
    Node->Children[oct] = calloc(1, sizeof(struct OctreeNode));
    Node->Children[oct]->Parent = Node;
    Node->Children[oct]->HalfSize = Node->HalfSize / 2;
    Node->Children[oct]->Npart = 0;
    Node->Children[oct]->level = Node->level + 1;
    Node->Children[oct]->IsLeaf = 0;
    Node->Children[oct]->ID = *label;

    for (int i = 0; i < 8; i++) {
        Node->Children[oct]->ChildExists[i] = 0;
    }

    (*label)++;
}

int sign(double x) {
    if (x >= 0) return 1;
    return -1;
}

void octree(struct ketju_system *R, struct OctreeNode *Node,
            int MaxLeafSize, int *label) {

    int NpartOctant[8] = {0};
    int triplet[3];
    for (int i = 0; i < Node->Npart; i++) {
        int j = Node->ObjectList[i];
        for (int k = 0; k < 3; k++) {
            triplet[k] = sign(R->Pos[3 * j + k] - Node->Center[k]);
        }
        int oct = get_octant(triplet);

        if (NpartOctant[oct] == 0) {
            allocate_new_octree_node(Node, oct, label);
        }

        indexlist[oct][NpartOctant[oct]] = j;

        NpartOctant[oct]++;
        Node->Children[oct]->Npart++;
    }

    for (int oct = 0; oct < 8; oct++) {

        if (NpartOctant[oct] == 0) { continue; }

        Node->Children[oct]->ObjectList =
            calloc(Node->Children[oct]->Npart, sizeof(int));

        for (int i = 0; i < Node->Children[oct]->Npart; i++) {
            Node->Children[oct]->ObjectList[i] = indexlist[oct][i];
        }

        get_triplet_from_octant(oct, triplet);

        for (int k = 0; k < 3; k++) {
            Node->Children[oct]->Center[k] =
                Node->Center[k] + triplet[k] * Node->Children[oct]->HalfSize;
        }

        for (int i = 0; i < Node->Children[oct]->Npart; i++) {
            ParticleInNode[Node->Children[oct]->ObjectList[i]] =
                Node->Children[oct];
        }

        if (Node->Children[oct]->Npart == 1) {
            Node->Children[oct]->IsLeaf = 1;
        }
    }

    for (int oct = 0; oct < 8; oct++) {
        if (Node->ChildExists[oct]) {
            if (Node->Children[oct]->Npart > MaxLeafSize) {
                octree(R, Node->Children[oct], MaxLeafSize, label);
            } else {
                Node->Children[oct]->IsLeaf = 1;
            }
        }
    }
}

void spatialpartition(struct ketju_system *R, int MaxLeafSize) {

    double rmax = get_rmax(R);

    OctreeRootNode = calloc(1, sizeof(struct OctreeNode));

    OctreeRootNode->ID = 0;
    OctreeRootNode->Parent = NULL;
    OctreeRootNode->HalfSize = 1.001 * rmax;
    OctreeRootNode->Npart = R->NumVertex;
    OctreeRootNode->ObjectList = calloc(R->NumVertex, sizeof(int));
    OctreeRootNode->level = 0;
    for (int i = 0; i < 3; i++) {
        OctreeRootNode->Center[i] = 0.0;
    }
    for (int i = 0; i < 8; i++) {
        OctreeRootNode->ChildExists[i] = 0;
    }

    for (int i = 0; i < R->NumVertex; i++) {
        OctreeRootNode->ObjectList[i] = i;
    }

    int label = 1;
    octree(R, OctreeRootNode, MaxLeafSize, &label);
}

int get_longest(double w[3], int id[3]) {

    double L = -1;
    int ind = -1;
    for (int m = 0; m < 3; m++) {
        if (w[m] > L) {
            L = w[m];
            ind = m;
        }
    }
    return id[ind];
}

int get_edge(int Nleafs, int va, int vb) {

    int N = Nleafs - 2;
    int lo, hi;
    lo = va, hi = vb;
    if (va > vb) {
        lo = vb;
        hi = va;
    }
    int index = hi - 1;
    for (int p = 0; p < lo; p++) {
        index += N;
        N--;
    }
    return index;
}

void SolveMetaMST(struct ketju_system *R, int Nleafs, int *Npart,
                  int **Particles, int *NumEdge) {

    int Nedge = Nleafs * (Nleafs - 1) / 2;
    struct WeightIndex2 *W = calloc(Nedge, sizeof(struct WeightIndex2));
    int *InMst = calloc(Nleafs, sizeof(int));

    int CumNumEdge = 0, istart = 0;
#ifdef PARALLEL
    int ThisBlock, jstart;
    loop_scheduling_N2(Nleafs, Ntask, &istart, &jstart, &ThisBlock, &CumNumEdge,
                       ThisTask);
#endif

    struct OctreeNode *Node_i, *Node_j;
    int counter = 0, ind1 = -1, ind2 = -1;

    int *ind1s = calloc(Nedge, sizeof(int));
    int *ind2s = calloc(Nedge, sizeof(int));
    double *weights = calloc(Nedge, sizeof(double));

    counter = 0;
    for (int i = istart; i < Nleafs; i++) {

        Node_i = ParticleInNode[Particles[i][0]];

#ifdef PARALLEL
        if (i > istart) { jstart = i + 1; }
        for (int j = jstart; j < Nleafs; j++) {
#else
        for (int j = i + 1; j < Nleafs; j++) {
#endif

            Node_j = ParticleInNode[Particles[j][0]];

            double r2 = 0;
            for (int k = 0; k < 3; k++) {
                double ds = Node_i->Center[k] - Node_j->Center[k];
                r2 = ds * ds;
            }
            double NodeSeparation = sqrt(r2);
            double NodeSize =
                1.01 * sqrt(2) *
                (Node_i->HalfSize + Node_j->HalfSize); // with safety buffer

            if (NodeSeparation <= NodeSize) {

                double MinWeight = 1e10;
                for (int q = 0; q < Npart[i]; q++) {
                    for (int p = 0; p < Npart[j]; p++) {
                        double r2 = 0;
                        int i1 = Particles[i][q];
                        int i2 = Particles[j][p];
                        for (int k = 0; k < 3; k++) {
                            double ds = R->Pos[3 * i1 + k] - R->Pos[3 * i2 + k];
                            r2 += ds * ds;
                        }
                        double r = sqrt(r2);
                        if (r < MinWeight) {
                            MinWeight = r;
                            ind1 = i1;
                            ind2 = i2;
                        }
                    }
                }

            } else {

                double MinWeight = 1e10;
                for (int q = 0; q < Npart[i]; q++) {
                    int i1 = Particles[i][q];
                    double r2 = 0;
                    for (int k = 0; k < 3; k++) {
                        double ds = R->Pos[3 * i1 + k] - Node_j->Center[k];
                        r2 += ds * ds;
                    }
                    double r = sqrt(r2);
                    if (r < MinWeight) {
                        MinWeight = r;
                        ind1 = i1;
                    }
                }
                MinWeight = 1e10;
                for (int p = 0; p < Npart[j]; p++) {
                    int i2 = Particles[j][p];
                    double r2 = 0;
                    for (int k = 0; k < 3; k++) {
                        double ds = R->Pos[3 * i2 + k] - Node_i->Center[k];
                        r2 += ds * ds;
                    }
                    double r = sqrt(r2);
                    if (r < MinWeight) {
                        MinWeight = r;
                        ind2 = i2;
                    }
                }
            }

            r2 = 0;
            for (int k = 0; k < 3; k++) {
                double ds = R->Pos[3 * ind1 + k] - R->Pos[3 * ind2 + k];
                r2 += ds * ds;
            }
            double r = sqrt(r2);

            weights[CumNumEdge + counter] = r;
            ind1s[CumNumEdge + counter] = ind1;
            ind2s[CumNumEdge + counter] = ind2;
            counter++;

#ifdef PARALLEL
            if (counter == ThisBlock) {
                loop = 0;
                break;
            }
        }
        if (loop == 0) { break; }
#else
        }
#endif
    }

#ifdef PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, ind1s, Nedge, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ind2s, Nedge, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, weights, Nedge, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
#endif

    counter = 0;
    for (int i = 0; i < Nleafs; i++) {
        for (int j = i + 1; j < Nleafs; j++) {
            W[counter].weight = weights[counter];
            W[counter].id1 = ind1s[counter];
            W[counter].id2 = ind2s[counter];
            W[counter].index1 = i;
            W[counter].index2 = j;
            counter++;
        }
    }

    free(weights);
    free(ind1s);
    free(ind2s);

    istart = 0;
    int istop = Nedge;
#ifdef PARALLEL
    int block = (int)floor(Nedge / Ntask);
    istart = ThisTask * block;
    istop = (ThisTask + 1) * block;
    if (ThisTask == Ntask - 1) istop = Nedge;
#endif

    int *flag = calloc(Nedge, sizeof(int));

    int Noff = 0;
    double w[3];
    int ind[3];
    for (int e1 = istart; e1 < istop; e1++) {

        int v1 = W[e1].index1;
        int v2 = W[e1].index2;
        ind[0] = e1;
        w[0] = W[e1].weight;

        if (w[0] < 0) continue;

        for (int v3 = 0; v3 < Nleafs; v3++) {

            if (w[0] < 0) continue;

            if (v3 == v1 || v3 == v2) continue;

            ind[1] = get_edge(Nleafs, v1, v3);
            ind[2] = get_edge(Nleafs, v2, v3);

            w[1] = W[ind[1]].weight;
            w[2] = W[ind[2]].weight;
            if (w[1] < 0 || w[2] < 0) continue;
            int elong = get_longest(w, ind);

            flag[elong] = 1;
            W[elong].weight *= -1;
        }
    }

#ifdef PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, flag, Nedge, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

    for (int e1 = 0; e1 < Nedge; e1++) {
        if (flag[e1] == 0) Noff++;
    }

    struct WeightIndex2 *Wfilter =
        calloc(Nedge - Noff, sizeof(struct WeightIndex2));

    counter = 0;
    for (int e1 = 0; e1 < Nedge; e1++) {
        if (flag[e1] == 0) {
            Wfilter[counter].weight = W[e1].weight;
            Wfilter[counter].id1 = W[e1].id1;
            Wfilter[counter].id2 = W[e1].id2;
            Wfilter[counter].index1 = W[e1].index1;
            Wfilter[counter].index2 = W[e1].index2;
            counter++;
        }
    }

    Nedge -= Noff;
    free(W);
    free(flag);

    qsort(Wfilter, Nedge, sizeof(struct WeightIndex2), cmp_weight_index2);

    InMst[0] = 1;

    int remaining = Nleafs - 1;
    istart = 0;
    istop = Nedge;
#ifdef PARALLEL
    block = (int)floor(Nedge / Ntask);
    istart = ThisTask * block;
    istop = (ThisTask + 1) * block;
    if (ThisTask == Ntask - 1) istop = Nedge;
#endif

    do {
        struct WeightIndex wi;
        wi.weight = 1e10;
        wi.index = -1;

        for (int i = istart; i < istop; i++) {
            int i1 = Wfilter[i].index1;
            int i2 = Wfilter[i].index2;
            if (InMst[i1] != InMst[i2]) {
                wi.index = i;
                wi.weight = Wfilter[i].weight;
                break;
            }
        }

#ifdef PARALLEL
        MPI_Allreduce(MPI_IN_PLACE, &wi, 1, MPI_DOUBLE_INT, MPI_MINLOC,
                      MPI_COMM_WORLD);
#endif

        InMst[Wfilter[wi.index].index1] = 1;
        InMst[Wfilter[wi.index].index2] = 1;

        R->EdgeInMST[*NumEdge].vertex1 = Wfilter[wi.index].id1;
        R->EdgeInMST[*NumEdge].vertex2 = Wfilter[wi.index].id2;
        R->EdgeInMST[*NumEdge].weight = Wfilter[wi.index].weight;
        (*NumEdge)++;

        remaining--;
    } while (remaining > 0);

    free(Wfilter);
    free(InMst);
}

void SolveSubMST(struct ketju_system *R, int Npart, int *Particles,
                 int label, int *NumEdge) {

    if (Npart == 1) return;

    int Nedge = Npart * (Npart - 1) / 2;
    struct WeightIndex2 *W = calloc(Nedge, sizeof(struct WeightIndex2));
    int *InMst = calloc(Npart, sizeof(int));

    int counter = 0;
    for (int i = 0; i < Npart; i++) {
        int ind1 = Particles[i];
        for (int j = i + 1; j < Npart; j++) {
            int ind2 = Particles[j];
            double r2 = 0;
            for (int k = 0; k < 3; k++) {
                double ds = R->Pos[3 * ind1 + k] - R->Pos[3 * ind2 + k];
                r2 += ds * ds;
            }
            double r = sqrt(r2);
            W[counter].weight = r;
            W[counter].index1 = i;
            W[counter].index2 = j;
            W[counter].id1 = ind1;
            W[counter].id2 = ind2;
            counter++;
        }
    }

    qsort(W, Nedge, sizeof(struct WeightIndex2), cmp_weight_index2);

    InMst[0] = 1;

    int remaining = Npart - 1;

    do {

        for (int i = 0; i < Nedge; i++) {
            int i1 = W[i].index1;
            int i2 = W[i].index2;
            if (InMst[i1] != InMst[i2]) {
                if (InMst[i1] == 1) {
                    InMst[i2] = 1;
                } else {
                    InMst[i1] = 1;
                }

                R->EdgeInMST[*NumEdge].vertex1 = W[i].id1;
                R->EdgeInMST[*NumEdge].vertex2 = W[i].id2;
                R->EdgeInMST[*NumEdge].weight = W[i].weight;
                (*NumEdge)++;
                break;
            }
        }

        remaining--;
    } while (remaining > 0);

    free(W);
    free(InMst);
}

void SwapEdge(struct ketju_system *R, int i, int j) {

    int v1 = R->EdgeInMST[i].vertex1;
    int v2 = R->EdgeInMST[i].vertex2;
    double w = R->EdgeInMST[i].weight;

    R->EdgeInMST[i].vertex1 = R->EdgeInMST[j].vertex1;
    R->EdgeInMST[i].vertex2 = R->EdgeInMST[j].vertex2;
    R->EdgeInMST[i].weight = R->EdgeInMST[j].weight;

    R->EdgeInMST[j].vertex1 = v1;
    R->EdgeInMST[j].vertex2 = v2;
    R->EdgeInMST[j].weight = w;
}

void CombineAllMST(struct ketju_system *R, int NumEdge) {

    int next = -1;
    double rmin = DBL_MAX;
    for (int i = 0; i < R->NumVertex; i++) {
        double r2 = 0;
        for (int k = 0; k < 3; k++) {
            double ds = R->Pos[3 * i + k];
            r2 += ds * ds;
        }
        if (rmin > sqrt(r2)) {
            rmin = sqrt(r2);
            next = i;
        }
    }

    int counter = 0;
    R->Vertex[next].inMST = 1;

    int istop = NumEdge;

    do {

        struct WeightIndex wi;
        wi.weight = 1e10;
        wi.index = -1;

        for (int i = counter; i < istop; i++) {
            int v1 = R->EdgeInMST[i].vertex1;
            int v2 = R->EdgeInMST[i].vertex2;
            if (R->Vertex[v1].inMST != R->Vertex[v2].inMST) {
                if (R->EdgeInMST[i].weight < wi.weight) {
                    wi.weight = R->EdgeInMST[i].weight;
                    wi.index = i;
                }
            }
        }

        int new = R->EdgeInMST[wi.index].vertex1;
        int old = R->EdgeInMST[wi.index].vertex2;
        if (R->Vertex[new].inMST == 1) {
            old = R->EdgeInMST[wi.index].vertex1;
            new = R->EdgeInMST[wi.index].vertex2;
        }

        R->Vertex[new].inMST = 1;
        R->Vertex[new].parent = old;
        R->Vertex[new].level = R->Vertex[old].level + 1;
        R->Vertex[new].edgetoparent = counter;

        for (int k = 0; k < 3; k++) {
            R->State[3 * counter + k] =
                R->Pos[3 * new + k] - R->Pos[3 * old + k];
            R->State[3 * (R->NumVertex - 1) + 3 * counter + k] =
                R->Vel[3 * new + k] - R->Vel[3 * old + k];
        }

        SwapEdge(R, counter, wi.index);
        counter++;

    } while (counter < NumEdge);
}

void DivideAndConquerPrim(struct ketju_system *R) {

    my_barrier();
    clock_t t0 = clock();

    for (int i = 0; i < R->NumVertex; i++) {
        R->Vertex[i].id = i;
        R->Vertex[i].degree = 0;
        R->Vertex[i].edgetoparent = -1;
        R->Vertex[i].parent = -1;
        R->Vertex[i].level = 0;
        R->Vertex[i].inMST = 0;
    }

    int Nleafs = 0;
    int MaxLeafSize = (int)floor(3 * sqrt(R->NumVertex));

    clock_t t10 = clock();
    spatialpartition(&R[0], MaxLeafSize);

    GetNumberOfLeafs(OctreeRootNode, &Nleafs);
    int *LeafSizes = calloc(Nleafs, sizeof(int));
    int **LeafParticles = calloc(Nleafs, sizeof(int *));
    clock_t t11 = clock();
    tprim1 += (double)(t11 - t10) / CLOCKS_PER_SEC;

    clock_t t20 = clock();
    Nleafs = 0;
    list_octree_level(OctreeRootNode, LeafSizes, LeafParticles, &Nleafs);
    clock_t t21 = clock();
    tprim2 += (double)(t21 - t20) / CLOCKS_PER_SEC;

    clock_t t30 = clock();
    int NumEdge = 0;
    for (int i = 0; i < Nleafs; i++) {
        SolveSubMST(R, LeafSizes[i], LeafParticles[i], i, &NumEdge);
    }
    clock_t t31 = clock();
    tprim3 += (double)(t31 - t30) / CLOCKS_PER_SEC;

    clock_t t40 = clock();
    SolveMetaMST(R, Nleafs, LeafSizes, LeafParticles, &NumEdge);
    clock_t t41 = clock();
    tprim4 += (double)(t41 - t40) / CLOCKS_PER_SEC;

    clock_t t50 = clock();
    CombineAllMST(R, NumEdge);
    clock_t t51 = clock();
    tprim5 += (double)(t51 - t50) / CLOCKS_PER_SEC;

    free_octree(OctreeRootNode);
    for (int i = 0; i < Nleafs; i++) {
        free(LeafParticles[i]);
    }
    free(LeafParticles);
    free(LeafSizes);

    clock_t t1 = clock();
    time_mst += (double)(t1 - t0) / CLOCKS_PER_SEC;
}
#endif
