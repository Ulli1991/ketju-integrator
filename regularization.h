//#define SERIAL
#define PARALLEL
//#define PARALLEL_QSORT

//#define CHAIN

#define GLOBAL_ND 2
#define GCONST 43007.1
#define KMAX 8
#define MAXSORTTASK 8

#define M_PI 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214
#define SPEEDOFLIGHT 299792.0

// structs

#undef USE_PN
//#define USE_PN

enum velocity_type { PHYSICAL, AUXILIARY };
enum variable_type { POSITION, VELOCITY, OTHER };


enum timing_type {
CPU_TIME_MISC,
CPU_TIME_ACC,
CPU_TIME_POTENTIALENERGY,
CPU_TIME_PRIM,
CPU_TIME_GBS
};

clock_t *cpu_times;
clock_t previous_time, current_time;


int Ntask;
int ThisTask;

int Niter;
int Nredo;
int MyC;
clock_t wallclock[10];

int NumGbsGroup;
int NumTaskPerGbsGroup;
int ThisGbsGroup;
int ThisTask_in_GbsGroup;

int NumTaskPerSortGroup;
int NumSortGroup;
int ThisSortGroup;
int ThisTask_in_SortGroup;

struct OctreeNode {
    int ChildExists[8];
    struct OctreeNode *Children[8];
    struct OctreeNode *Parent;
    double      Center[3];
    double      HalfSize;
    int *ObjectList;
    int Npart;
    int level;
    int IsLeaf;
    int ID;
} *OctreeRootNode, **ParticleInNode;


struct WeightIndex {
        double weight;
        int index;
};
struct WeightIndex2 {
        double weight;
        int index1;
	int index2;
	int id1;
	int id2;
};

struct GraphEdge {
        int id;
        int vertex1;
        int vertex2;
        float weight;
};

struct GraphVertex {
        int id;
        int degree;
        int type;
        int parent;
        int edgetoparent;
        int level;
        int inMST;
	int cell[3];
};

struct RegularizedRegion {

        int NumVertex;
        int NumEdge;
	int NumDynVariables;

        double *Pos;
        double *Vel;
        double *Mass;
        double *Acc;
        double *InitialState;
        double *State;
        double *MSTedgeAcc;
        double *AuxVel;
        double *AuxEdgeVel;
	double *AccPN;
        double *MSTedgeAcc_PN;

	struct GraphEdge *Edge;
        struct GraphEdge *LocalEdge;
	struct GraphEdge *LocalEdgeSubset;
        struct GraphVertex *Vertex;
        int *MSTedgeList;
	struct GraphEdge *EdgeInMST;

        double T;
        double U;
        double H;
        double B;
        double time;

	double Udot;

	int BIndex;
	int TimeIndex;

        double CoM_Pos[3];
        double CoM_Vel[3];

	// relative error tolerance
	double gbs_tolerance;
	double output_time_tolerance;

	int korder;

	double Hstep;

};

struct ComTask {
	int GroupToPerformTask;
	struct RegularizedRegion *ThisR;
	int WhichRegion;
	int NumberOfSubsteps;
	double *ResultState;
};

struct ToDoList {
	int NumberOfComputationalTasks;
	struct ComTask *ComputationalTask;
	struct RegularizedRegion *CopyOfSingleRegion;
} ComputationToDoList;

double E0;

// functions

void compute_Post_Newtonian_Acc(struct RegularizedRegion *R, double *Vel);
int check_relative_proximity( int v1, int v2, const int Nd, struct RegularizedRegion *R, int *d, int *path, int *sign);
