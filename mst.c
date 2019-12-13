#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <limits.h>
#include "regularization.h"
#include <string.h>
#include <unistd.h>

//#undef OLDPRIM
#define OLDPRIM

#define mydouble double

int **indexlist;
int extrapolate_all_variables( int NumberOfSystems, int korder );
int Nsearch;

double GBSTOL;

double loopt1,loopt2,loopt3,loopt4;

double *errorbuf;
double *resultbuf;
double *y;
double *yscal;
double *maxerror;
double **error,**result;


double **gbsS;
double *allerrors;
double total_memory;

int MAXPART;
double ttlratio[8];

#ifdef PARALLEL
#include <mpi.h>
#endif

#ifdef PARALLEL
MPI_Comm MPI_COMM_GBS_GROUP;
#endif
#ifdef PARALLEL_QSORT
MPI_Comm MPI_COMM_SORT;
#endif


int *fastindex;
int *fastv1;
int *fastv2;
double *fastweight;


int primfail;
int primsucc;
double sum_order;

double tprim1,tprim2,tprim3,tprim4,tprim5,tprim6,tprim7;
double text1,text2,text3;

double time_edge;
double time_coord1;
double time_coord2;
double time_coord3;
double time_coord4;
double time_force;
double time_comm;
double time_mst;
double time_gbs;
double time_gbscomm;

double safety;
double finalerror_energy;
double realtime;
double fictitioustime;

int ok_steps;
int failed_steps;
double EN0,Lz0,a0,e0,theta0;
double EN1,Lz1,a1,e1,theta1;

FILE *output;
clock_t wtime1,wtime0;
clock_t msttime1,msttime0;

void my_barrier(void){

#ifdef PARALLEL
        MPI_Barrier(MPI_COMM_WORLD);
#endif

}

void die(void){

	my_barrier();
	if(ThisTask==0) printf("The code is dead because 'die()' was called..\n");
#ifdef PARALLEL
        MPI_Finalize();
#endif
        exit(0);

}

void measure_cpu_time( enum timing_type index ){
	current_time = clock();
	cpu_times[index] += current_time-previous_time;
	previous_time = current_time;	
}

void analysis_for_smallest_binary_period( struct RegularizedRegion *R ){

        int N = R->NumVertex;
        double min_a = DBL_MAX;
        double min_P;


	double vvec[3]={0},vvec2[3]={0};
	for(int i=0;i<N;i++){
		for(int k=0;k<3;k++){
			vvec[k]  += R->Vel[3*i+k]/N;
			vvec2[k] += R->Vel[3*i+k]*R->Vel[3*i+k]/N;
		}
	}
	double sigmas[3],sigma=0;
	for(int k=0;k<3;k++){
		sigmas[k] = sqrt(vvec2[k]-vvec[k]*vvec[k]);
		sigma = sigmas[k]*sigmas[k];
	}
	sigma = sqrt(sigma);

	int Nhard = 0;
        for(int i=0;i<N;i++){
		for(int j=i+1;j<N;j++){
                	double dr[3],dv[3],r2=0,r,v2=0;
                	for(int k=0;k<3;k++){
                        	dr[k]   = R->Pos[3*j+k] - R->Pos[3*i+k];
                        	dv[k]   = R->Vel[3*j+k] - R->Vel[3*i+k];
                        	r2     += dr[k]*dr[k];
                        	v2     += dv[k]*dv[k];
                	}
                	r = sqrt(r2);

                	double mi = R->Mass[i];
			double mj = R->Mass[j];
			double M  = mi+mj;
			double mu = (mi*mj)/M;

			double E = 0.5*mu*v2 - GCONST*mu*M/r;
			double a = -GCONST*mu*M/(2*E);

			if( E>0 ){
				continue;
			}

			double B =  -E;

			double meanEkin = mi*sigma*sigma;
			if( B > meanEkin ){
				Nhard++;
			}

			// printf("%d %d %e %e %e\n",i,j,a,B,meanEkin);

                	if( a < min_a && a>0 ){
                        	min_a = a;
                        	min_P = sqrt( pow(min_a,3)/(GCONST*M) );
                	}
	        }
	}

if(ThisTask==0){
        printf("%d %.10e %.10e %lf %d\n",MAXPART, min_a, min_P, sigma, Nhard );
}
	//die();

}


void analysis_for_smallest_period_around_bh( struct RegularizedRegion *R ){

	int N = R->NumVertex;
	int bhindex = N-1;
	double bhpos[3], bhvel[3];
	double Mbh = R->Mass[bhindex];
	for(int k=0;k<3;k++){
		bhpos[k] = R->Pos[3*bhindex+k];
		bhvel[k] = R->Vel[3*bhindex+k];
	}

	double min_a = DBL_MAX;
	double min_P;
	for(int i=0;i<bhindex;i++){
		double dr[3],dv[3],r2=0,r,v2=0;
		for(int k=0;k<3;k++){
			dr[k]   = R->Pos[3*i+k] - bhpos[k];
			dv[k]   = R->Vel[3*i+k] - bhvel[k];
			r2     += dr[k]*dr[k];
			v2     += dv[k]*dv[k];
		}
		r = sqrt(r2);
		double mstar = R->Mass[i];
		double GM = GCONST*(Mbh+mstar);
		double a = 1.0/(2.0/r-v2/GM);

		if( a < min_a && a>0 ){
			min_a = a;
			min_P = sqrt( pow(min_a,3)/GM );
		}

	}
	printf("%.10e,%.10e %lf\n",min_a,min_P,1e-4*0.9785/min_P);
}

void analysis_for_output( struct RegularizedRegion *R, double *a, double *e, double *theta, double *EN, double *Lz ){

                double dr,r=0,v2=0,dv,rdotv=0;
                double rvec[3],vvec[3];
                double Ekin=0,Epot=0;

                for(int k=0;k<3;k++){
                        dr = R->Pos[3*0+k]-R->Pos[3*1+k];
                        rvec[k] = dr;
                        r += dr*dr;
                        dv = R->Vel[3*0+k]-R->Vel[3*1+k];
                        vvec[k] = dv;
                        v2 += dv*dv;
                        rdotv += dr*dv;
                        Ekin += 0.5*R->Mass[0]*R->Vel[3*0+k]*R->Vel[3*0+k] + 0.5*R->Mass[1]*R->Vel[3*1+k]*R->Vel[3*1+k];
                }
                r = sqrt(r);

                Epot = -GCONST*( R->Mass[0]*R->Mass[1] )/r;
                double GM = GCONST*(R->Mass[0]+R->Mass[1]);

                *Lz = rvec[0]*vvec[1]-rvec[1]*vvec[0];
                *EN = Ekin+Epot;

                *a = 1.0/(2.0/r - v2/GM);

                double ecc[3];
                *e=0;
                for(int k=0;k<3;k++){
                        ecc[k] = (1.0/r-1.0/(*a))*rvec[k] - rdotv*vvec[k]/GM;
                        *e += ecc[k]*ecc[k];
                }
                *e = sqrt(*e);

                double Kx = ( v2/GM - 1.0/r ) * rvec[0] - rdotv/GM * vvec[0];
                double Ky = ( v2/GM - 1.0/r ) * rvec[1] - rdotv/GM * vvec[1];
                *theta = atan2(Ky,Kx);
}

int cmp_weight_index2(const void *a, const void *b){
    struct WeightIndex2 *a1 = (struct WeightIndex2 *)a;
    struct WeightIndex2 *a2 = (struct WeightIndex2 *)b;
    if ((*a1).weight > (*a2).weight)
        return 1;
    else if ((*a1).weight < (*a2).weight)
        return -1;
    else
        return 0;
}


int cmp_weight_index(const void *a, const void *b){
    struct WeightIndex *a1 = (struct WeightIndex *)a;
    struct WeightIndex *a2 = (struct WeightIndex *)b;
    if ((*a1).weight > (*a2).weight)
        return 1;
    else if ((*a1).weight < (*a2).weight)
        return -1;
    else
        return 0;
}


int cmp(const void *a, const void *b){
    struct GraphEdge *a1 = (struct GraphEdge *)a;
    struct GraphEdge *a2 = (struct GraphEdge *)b;
    if ((*a1).weight > (*a2).weight)
        return 1;
    else if ((*a1).weight < (*a2).weight)
        return -1;
    else
        return 0;
}

int cmp_int( const void* a, const void* b){
     int int_a = * ( (int*) a );
     int int_b = * ( (int*) b );
     if ( int_a == int_b ) return 0;
     else if ( int_a < int_b ) return -1;
     else return 1;
}

int cmp_double( const void* a_in, const void* b_in){
     double a = * ( (double*) a_in );
     double b = * ( (double*) b_in );

}

#ifdef PARALLEL_QSORT

void parallel_qsort_final_level( struct WeightIndex *W, int Nitem, struct WeightIndex *SortedW, int *SortedNitem, int TotalNitem ){

	qsort( W, Nitem, sizeof(struct WeightIndex), cmp_weight_index );

	int *Nitems = calloc(NumTaskPerSortGroup,sizeof(int));

	Nitems[ThisTask] = Nitem;

	MPI_Allreduce(MPI_IN_PLACE,Nitems,NumTaskPerSortGroup,MPI_INT,MPI_SUM,MPI_COMM_SORT);

	int istart,istop;
	int CumNitem = 0;

	for(int i=0;i<NumTaskPerSortGroup;i++){
		if(ThisTask==i) istart = CumNitem;
		CumNitem += Nitems[i];
		if(ThisTask==i) istop  = CumNitem;
	}

	double *Wweight = calloc(TotalNitem,sizeof(double));
	int *Windex     = calloc(TotalNitem,sizeof(int));

	for(int i=0;i<Nitem;i++){
		Wweight[istart+i] = W[i].weight;
		Windex[istart+i]  = W[i].index;
	}

	MPI_Allreduce(MPI_IN_PLACE,Wweight,TotalNitem,MPI_DOUBLE,MPI_SUM,MPI_COMM_SORT);
	MPI_Allreduce(MPI_IN_PLACE,Windex, TotalNitem,MPI_INT,   MPI_SUM,MPI_COMM_SORT);

	for(int i=0;i<TotalNitem;i++){
		SortedW[i].weight = Wweight[i];
		SortedW[i].index  = Windex[i];
	}

	*SortedNitem = Nitem;

	free(Nitems);
	free(Wweight);
	free(Windex);
	free(W);
}

void parallel_qsort( struct WeightIndex *W, int Nitem, int NumSortTask, int ThisSortTask, int *ActualTaskNumbers, struct WeightIndex *SortedW, int *SortedNitem, int round, int TotalNitem ){

	if(Nitem==0) printf("sort: Nitem cannot be zero. Round = %d, TotalNitem = %d.\n",round,TotalNitem);

	double pivots[NumTaskPerSortGroup];
	for(int i=0;i<NumTaskPerSortGroup;i++){
		pivots[i] = 0;
	}

	if( ThisTask == ActualTaskNumbers[0] ){
		int index_pivot = rand()%(Nitem+1);
		pivots[ThisTask] = W[index_pivot].weight;
	}

	MPI_Allreduce( MPI_IN_PLACE, pivots, NumTaskPerSortGroup, MPI_DOUBLE, MPI_SUM, MPI_COMM_SORT );
	double pivot = pivots[ThisTask-ThisSortTask];

	struct WeightIndex *Lo = calloc(10*Nitem,sizeof(struct WeightIndex));
	struct WeightIndex *Hi = calloc(10*Nitem,sizeof(struct WeightIndex));

	int Nlo = 0, Nhi = 0;
	for(int i=0;i<Nitem;i++){

		if( W[i].weight < pivot ){
			Lo[Nlo].weight = W[i].weight;
			Lo[Nlo].index  = W[i].index;
			Nlo++;
		} else {
			Hi[Nhi].weight = W[i].weight;
                        Hi[Nhi].index  = W[i].index;
                        Nhi++;
		}
	}

	free(W);


	int Nsend=0,Nrecv=0,from=-1,to=0;
	if(ThisSortTask<NumSortTask/2){
		Nsend = Nhi;
		to    = ActualTaskNumbers[ ThisSortTask + NumSortTask/2 ];
		from  = to;
	} else {
		Nsend = Nlo;
		to    = ActualTaskNumbers[ ThisSortTask - NumSortTask/2 ];
                from  = to;
	}

	if(ThisSortTask<NumSortTask/2){
		MPI_Send(&Nsend, 1, MPI_INT, to,   0, MPI_COMM_SORT);
	} else {
		MPI_Recv(&Nrecv, 1, MPI_INT, from, 0, MPI_COMM_SORT,MPI_STATUS_IGNORE);
	}

	if(ThisSortTask>=NumSortTask/2){
                MPI_Send(&Nsend, 1, MPI_INT, to,   0, MPI_COMM_SORT);
        } else {
                MPI_Recv(&Nrecv, 1, MPI_INT, from, 0, MPI_COMM_SORT,MPI_STATUS_IGNORE);
        }

	if(ThisSortTask>=NumSortTask/2){
		if(Nhi+Nrecv>10*Nitem){
			printf("Too many numbers by fac %lf\n", (1.0*Nhi+1.0*Nrecv)/(10*Nitem) );
		}
	} else {
		if(Nlo+Nrecv>10*Nitem){
                        printf("Too many numbers by fac %lf\n", (1.0*Nlo+1.0*Nrecv)/(10*Nitem));
                }
	}

	if(ThisSortTask<NumSortTask/2){
if(Nsend>0)                 MPI_Send(Hi, 	   Nsend, MPI_DOUBLE_INT, to,   0, MPI_COMM_SORT);
        } else {
if(Nrecv>0)                 MPI_Recv(&Hi[Nhi], Nrecv, MPI_DOUBLE_INT, from, 0, MPI_COMM_SORT,MPI_STATUS_IGNORE);
        }
	if(ThisSortTask>=NumSortTask/2){
if(Nsend>0)                 MPI_Send(Lo,       Nsend, MPI_DOUBLE_INT, to,   0, MPI_COMM_SORT);
        } else {
if(Nrecv>0)                 MPI_Recv(&Lo[Nlo], Nrecv, MPI_DOUBLE_INT, from, 0, MPI_COMM_SORT,MPI_STATUS_IGNORE);
        }

	int NextActualTaskNumbers[NumSortTask/2];

	if(NumSortTask>-1){

		if(ThisSortTask>=NumSortTask/2){
			Nlo  = 0;
			free(Lo);
			Nhi += Nrecv;
			for(int i=0;i<NumSortTask/2;i++){
                        	NextActualTaskNumbers[i] = ActualTaskNumbers[i+NumSortTask/2];
                	}
        	} else {
			Nhi  = 0;
			free(Hi);
			Nlo += Nrecv;
			for(int i=0;i<NumSortTask/2;i++){
				NextActualTaskNumbers[i] = ActualTaskNumbers[i];
			}
        	}	

	}

#if 1
	if( NumSortTask>2 ){

		if(ThisSortTask<NumSortTask/2){
			parallel_qsort( Lo, Nlo, NumSortTask/2, ThisSortTask, NextActualTaskNumbers, SortedW, SortedNitem, round+1, TotalNitem );
		} else {
			parallel_qsort( Hi, Nhi, NumSortTask/2, ThisSortTask-NumSortTask/2, NextActualTaskNumbers, SortedW, SortedNitem, round+1, TotalNitem );
		}

	} else {

		if(ThisSortTask<NumSortTask/2){
	                parallel_qsort_final_level( Lo, Nlo, SortedW, SortedNitem, TotalNitem );
		} else {
			parallel_qsort_final_level( Hi, Nhi, SortedW, SortedNitem, TotalNitem );
		}
                return;

	}
#endif


}

void run_parallel_qsort( struct WeightIndex *W, int Nitem){

	my_barrier();
	if(ThisTask==0) printf("start: %d\n", Nitem);

	double *Wweight = calloc(Nitem,sizeof(double));
        int    *Windex  = calloc(Nitem,sizeof(int));

	srand((unsigned) 2019);

	if( ThisSortGroup == 0 ){

		int SortedNitem = 0;
        	
		struct WeightIndex *ThisW    = calloc(Nitem,sizeof(struct WeightIndex));		
		struct WeightIndex *SortedW  = calloc(Nitem,sizeof(struct WeightIndex));

		int block  = Nitem/NumTaskPerSortGroup;
		int istart = ThisTask*block;
		int istop  = (ThisTask+1)*block;

		if(ThisTask==NumTaskPerSortGroup-1) istop = Nitem; 
		int ThisNitem = 0;

		for(int i=istart;i<istop;i++){
			ThisW[ThisNitem].weight = W[i].weight;
			ThisW[ThisNitem].index  = W[i].index;
			ThisNitem++;
		}

		int ActualTaskNumbers[NumTaskPerSortGroup];
		for(int i=0;i<NumTaskPerSortGroup;i++){
			ActualTaskNumbers[i] = i;
		}

		//parallel_qsort( ThisW, ThisNitem, NumTaskPerSortGroup, ThisTask, ActualTaskNumbers, SortedW, &SortedNitem, 0, Nitem);

		if(ThisTask==0){
			for(int i=0;i<Nitem;i++){
                		Wweight[i] = SortedW[i].weight;
                		Windex[i]  = SortedW[i].index;
	        	}
		}

		free(SortedW);

	}

	my_barrier();
        if(ThisTask==0) printf("before comm\n");
	my_barrier();

        MPI_Allreduce(MPI_IN_PLACE,Wweight,Nitem,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE,Windex, Nitem,MPI_INT,   MPI_SUM,MPI_COMM_WORLD);

	my_barrier();
        if(ThisTask==0) printf("before comm\n");
        my_barrier();

        for(int i=0;i<Nitem;i++){
                W[i].weight = Wweight[i];
                W[i].index  = Windex[i];
        }

        free(Wweight);
        free(Windex);

	my_barrier();
	if(ThisTask==0) printf("stop\n");

}
#endif

#ifdef PARALLEL_QSORT
void initialize_sort_comm(void){

        MPI_Group WORLD_GROUP, SORT_GROUP;
        MPI_Comm_group(MPI_COMM_WORLD, &WORLD_GROUP);
        int *myranks;

	NumTaskPerSortGroup   = (int) MAXSORTTASK;
        ThisSortGroup         = (int) floor(ThisTask/NumTaskPerSortGroup);
	NumSortGroup	      = (int) floor(Ntask/NumTaskPerSortGroup)+1;
        ThisTask_in_SortGroup = ThisTask%NumTaskPerSortGroup;
	if(ThisSortGroup==NumSortGroup-1) NumTaskPerSortGroup -= (NumSortGroup*NumTaskPerSortGroup-Ntask);

        myranks = calloc(NumTaskPerSortGroup, sizeof(int));

        for(int i=0;i<NumTaskPerSortGroup;i++){
                myranks[i] = ThisTask - ThisTask_in_SortGroup + i;
        }
        for(int i=0;i<NumSortGroup;i++){
                if(i == ThisSortGroup){
                        MPI_Group_incl(WORLD_GROUP, NumTaskPerSortGroup, myranks, &SORT_GROUP);
                }
        }

        MPI_Comm_create(MPI_COMM_WORLD, SORT_GROUP, &MPI_COMM_SORT);
        MPI_Group_free(&SORT_GROUP);
        free(myranks);

        MPI_Group_free(&WORLD_GROUP);

}
#endif


#ifdef PARALLEL
void initialize_node_comm(void){

	MPI_Group WORLD_GROUP, GBS_GROUP;
	MPI_Comm_group(MPI_COMM_WORLD, &WORLD_GROUP);
	int *myranks;

	ThisGbsGroup         = (int) floor(ThisTask/NumTaskPerGbsGroup);
	ThisTask_in_GbsGroup = ThisTask%NumTaskPerGbsGroup;

        myranks = calloc(NumTaskPerGbsGroup, sizeof(int));

        for(int i=0;i<NumTaskPerGbsGroup;i++){
                myranks[i] = ThisTask - ThisTask_in_GbsGroup + i;
	}
	for(int i=0;i<NumGbsGroup;i++){
		if(i == ThisGbsGroup){
        		MPI_Group_incl(WORLD_GROUP, NumTaskPerGbsGroup, myranks, &GBS_GROUP);
	    	}
    	}

    	MPI_Comm_create(MPI_COMM_WORLD, GBS_GROUP, &MPI_COMM_GBS_GROUP);
    	MPI_Group_free(&GBS_GROUP);
    	free(myranks);

    	MPI_Group_free(&WORLD_GROUP);
    	MPI_Comm_size(MPI_COMM_GBS_GROUP, &NumTaskPerGbsGroup);
    	MPI_Comm_rank(MPI_COMM_GBS_GROUP, &ThisTask_in_GbsGroup);

}
#endif

void initialize_mpi_or_serial(void){

// check for reasonable input
#if defined(PARALLEL) && defined(SERIAL)
	printf("Please select EITHER parallel OR serial. Currently both selected.\n");
	exit(0);
#endif
#if !defined(PARALLEL) && !defined(SERIAL)
        printf("Please select EITHER parallel OR serial. Currently none selected.\n");
        exit(0);
#endif

// Init MPI if needed
#ifdef PARALLEL
	MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &Ntask);
        MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

	NumGbsGroup        = 4;	//antti
        NumTaskPerGbsGroup = (int) Ntask/NumGbsGroup;

	if( NumGbsGroup*NumTaskPerGbsGroup != Ntask ){
		if(ThisTask==0){
			printf("NumGbsGroup*NumTaskPerGbsGroup != Ntask.        NumGbsGroup = %d, NumTaskPerGbsGroup = %d.\n",NumGbsGroup,NumTaskPerGbsGroup);
			exit(0);
		}
	}
	if( NumTaskPerGbsGroup>1 ){
		initialize_node_comm();
	} else {
		ThisGbsGroup = ThisTask;
		ThisTask_in_GbsGroup = 0;
	}
#else
	NumGbsGroup  = 1;
	Ntask        = 1;
	ThisTask     = 0;
	ThisGbsGroup = 0;
	ThisTask_in_GbsGroup = 0;
	NumTaskPerGbsGroup   = 1;
#endif

#ifdef PARALLEL_QSORT
	initialize_sort_comm();
#endif

}

int check_relative_proximity_ND_2( int v1, int v2, struct RegularizedRegion *R, int *d, int *path, int *sign ){

	int diff = abs( R->Vertex[v1].level - R->Vertex[v2].level );

        if( diff == 1 ){
                if( R->Vertex[v2].parent == v1 ){
                        *d = 1;
                        sign[0] = -1;
                        path[0] = R->Vertex[v2].edgetoparent;
                        return 1;
                } else if( R->Vertex[v1].parent == v2 ){
                        *d = 1;
                        sign[0] = +1;
                        path[0] = R->Vertex[v1].edgetoparent;
                        return 1;
                }
                return 0;
        } else if ( diff == 2 ){
                if( R->Vertex[ R->Vertex[v2].parent ].parent == v1 ){
                        *d = 2;
                        sign[0] = -1;   sign[1] = -1;
                        path[0] = R->Vertex[v2].edgetoparent;
                        path[1] = R->Vertex[ R->Vertex[v2].parent ].edgetoparent;
                        return 1;
                } else if( R->Vertex[ R->Vertex[v1].parent ].parent == v2 ){
                        *d = 2;
                        sign[0] = +1;   sign[1] = +1;
                        path[0] = R->Vertex[v1].edgetoparent;
                        path[1] = R->Vertex[ R->Vertex[v1].parent ].edgetoparent;
                        return 1;
                }
                return 0;
        } else if( diff == 0 ){
		if( R->Vertex[v1].parent == R->Vertex[v2].parent ){
			*d = 2;		
			sign[0] = -1;
			sign[1] = +1;
			path[0] = R->Vertex[v2].edgetoparent;
			path[1] = R->Vertex[v1].edgetoparent;
			return 1;
		}
		return 0;
	} 

	return 0;
}

int check_relative_proximity( int v1, int v2, const int Nd, struct RegularizedRegion *R, int *d, int *path, int *sign ){

	for(int i=0;i<Nd;i++){
		path[i] = -666;
		sign[i] = 0;
	}

	int proximity = 0;
	if( R->Vertex[v1].level == R->Vertex[v2].level ){

		if(R->Vertex[v1].level == 0 ){
			printf("This level stuff should not happen\n");
			printf("thistask: %d --- v1 v2 %d %d ---levels: %d %d\n", ThisTask,R->Vertex[v1].id, R->Vertex[v2].id,R->Vertex[v1].level, R->Vertex[v2].level);
			exit(0);
		}

		int halfNd = (int) floor(0.5*Nd);
	        int *pathend = calloc(Nd,sizeof(int));	total_memory += Nd*sizeof(int);

		for(int i=0;i<halfNd;i++){
			sign[i] = +1;
			path[i]	   = R->Vertex[v1].edgetoparent;
                        pathend[i] = R->Vertex[v2].edgetoparent;
			v1 = R->Vertex[v1].parent;
			v2 = R->Vertex[v2].parent;
			if(v1==v2){
				*d = 2*(i+1);
				int c=0;
				for(int m=i+1;m<(*d);m++){
                                        sign[m] = -1;
                                        path[m] = pathend[i-c];
					c++;
                                }
				proximity = 1;
				break;
			}
		}

		free(pathend);		total_memory -= Nd*sizeof(int);

	} else {
		*d = abs(R->Vertex[v1].level - R->Vertex[v2].level);
		if( *d <= Nd ){
			int lo,hi,thissign;
			if( R->Vertex[v1].level > R->Vertex[v2].level ){
				lo = v2;
				hi = v1;
				thissign = +1;
			} else {
				lo = v1;
                        	hi = v2;
				thissign = -1;
			}
			for(int i=0;i<*d;i++){
				path[i] = R->Vertex[hi].edgetoparent;
				sign[i] = thissign;
                        	hi = R->Vertex[hi].parent;
			}
			if(lo==hi){
                        	proximity = 1;
			}
		}
	}

	return proximity;
}

void loop_scheduling_N2( int Nrow, int Nt, int *istart, int *jstart, int *ThisBlock, int *CumNum, int task ){
	int Np = (Nrow*(Nrow-1))/2;
	int block = (int) floor(Np/Nt);
        int limit = (block+1)*Nt - Np;
	int ThisCumNum;
	if(task < limit){
                ThisCumNum = task*block;
                *ThisBlock = block;
        } else {
                ThisCumNum = limit*block + (task-limit)*(block+1);
                *ThisBlock = block+1;
        }

	double drow = 1.0*Nrow - 0.5 - 0.5*sqrt( (1.0-2.0*Nrow)*(1.0-2.0*Nrow) - 8.0*ThisCumNum );
        int row = (int) floor( drow );
        int rowstart = (int) round(-0.5*(1.0*row+1.0)*(row-2.0*Nrow)-1.0*Nrow);
        *istart = row;
        *jstart = ThisCumNum - rowstart + row + 1;
	*CumNum = ThisCumNum;

}

void get_v1_v2_fast( struct RegularizedRegion *R, int i, int *v1, int *v2 ){
	*v1 = R->EdgeInMST[i].vertex1;
	*v2 = R->EdgeInMST[i].vertex2;
}

void get_v1_v2( int N, int target, int *v1, int *v2){

	int final_element_on_row = 0;
	int elements_on_this_row;

	for(int row=0;row<N-1;row++){

		if(row == 0){
			elements_on_this_row = N-2-row;
		} else {
			elements_on_this_row = N-1-row;
		}

		final_element_on_row += elements_on_this_row;

		if(target <= final_element_on_row){
			*v1 = row;
			int diff = final_element_on_row - target;
			*v2 = N-1 - diff;
			break;
		}
	}

}


int sign(double x){
	if (x >= 0) return 1;
	return -1;
}

double get_rmax( struct RegularizedRegion *R ){
	double rmax2=0;
        for(int i=0;i<R->NumVertex;i++){
		double ds;
		double r2 = 0;
		for(int k=0;k<3;k++){
			ds = R->Pos[3*i+k];
			r2 += ds*ds;
		}
		if(r2>rmax2){
			rmax2=r2;
		}
	}
	return sqrt(rmax2);
}

void get_triplet( int index, int *triplet ){
	if( index>=100){
		triplet[2] = 1;
		index -= 100;
		if( index>=10){
			triplet[1] = 1;
			index -= 10;
			if( index == 1){
				triplet[0] = 1;
			} else {
				triplet[0] = -1;
			}
		} else {
			triplet[1] = 0;
			if( index == 1){
                                triplet[0] = 1;
                        } else {
                                triplet[0] = -1;
                        }
		}
	} else {
		triplet[2] = 0;
		if( index>=10){
                        triplet[1] = 1;
                        index -= 10;
			if( index == 1){
                                triplet[0] = 1;
                        } else {
                                triplet[0] = -1;
                        }
                } else {
                        triplet[1] = 0;
			if( index == 1){
                                triplet[0] = 1;
                        } else {
                                triplet[0] = -1;
                        }
                }
	}

}

int get_triplet_index( int num ){
	if(num==1) return 1;
	return 0;
}

int get_octant( int triplet[3] ){
        if( triplet[0]==1 ){
                if( triplet[1]==1){
			if( triplet[2] == -1){
                                return 6;
                        } else {
                                return 7;
                        }
                } else {
			if( triplet[2] == -1){
                                return 4;
                        } else {
                                return 5;
                        }
                }
        } else {
                if( triplet[1]==1){
                        if( triplet[2] == -1){
                                return 2;
                        } else {
                                return 3;
                        }
                } else {
                        if( triplet[2] == -1){
                                return 0;
                        } else {
                                return 1;
                        }
                }
        }
	return -1;
}

void get_triplet_from_octant( int octant, int *triplet ){

	if(octant==0){
		triplet[0] = -1; triplet[1] = -1; triplet[2] = -1;
		return;
	}
	if(octant==1){
                triplet[0] = -1; triplet[1] = -1; triplet[2] = +1;
                return;
        }
	if(octant==2){
                triplet[0] = -1; triplet[1] = +1; triplet[2] = -1;
                return;
        }
	if(octant==3){
                triplet[0] = -1; triplet[1] = +1; triplet[2] = +1;
                return;
        }
	if(octant==4){
                triplet[0] = +1; triplet[1] = -1; triplet[2] = -1;
                return;
        }
        if(octant==5){
                triplet[0] = +1; triplet[1] = -1; triplet[2] = +1;
                return;
        }
        if(octant==6){
                triplet[0] = +1; triplet[1] = +1; triplet[2] = -1;
                return;
        }
        if(octant==7){
                triplet[0] = +1; triplet[1] = +1; triplet[2] = +1;
                return;
        }
	return;
}

void list_octree_level( struct OctreeNode *Node, int *LeafSizes, int **LeafParticles, int *Nleafs ){

	if( Node->IsLeaf ){
		LeafSizes[*Nleafs] = Node->Npart;
		LeafParticles[*Nleafs] = calloc(Node->Npart, sizeof(int));
		for(int i=0;i<Node->Npart;i++){
			LeafParticles[*Nleafs][i] = Node->ObjectList[i];
		} 
		(*Nleafs)++;
	} else {
		for(int oct=0;oct<8;oct++){
			if( Node->Children[oct]->ChildExists ){
				list_octree_level( Node->Children[oct], LeafSizes, LeafParticles, Nleafs );
			}
		}
	}
}

void GetNumberOfLeafs( struct OctreeNode *Node, int *Nleaf ){
        if( Node->IsLeaf ){
                (*Nleaf)++;
        } else {
                for(int oct=0;oct<8;oct++){
                        if( Node->Children[oct]->ChildExists ){
                                GetNumberOfLeafs( Node->Children[oct], Nleaf );
                        }
                }
        }
}

void free_octree( struct OctreeNode *Node ){
        for(int oct=0;oct<8;oct++){
                if( Node->Children[oct]->ChildExists ){
                	free_octree( Node->Children[oct] );
        	}
	}
	free(Node->ObjectList);
	free(Node);
}


void allocate_new_octree_node( struct OctreeNode *Node, int oct, int *label ){

	Node->ChildExists[oct]          = 1;
	Node->Children[oct]             = calloc(1,sizeof(struct OctreeNode));
	Node->Children[oct]->Parent     = Node;
        Node->Children[oct]->HalfSize   = Node->HalfSize/2;
        Node->Children[oct]->Npart      = 0;
        Node->Children[oct]->level      = Node->level+1;
	Node->Children[oct]->IsLeaf 	= 0;
	Node->Children[oct]->ID	        = *label;

	for(int i=0;i<8;i++){
                Node->Children[oct]->ChildExists[i] = 0;
        }

	(*label)++;

}

void octree( struct RegularizedRegion *R, struct OctreeNode *Node, int MaxLeafSize, int *label ){

	int NpartOctant[8] = {0};
	int triplet[3];
	for(int i=0;i<Node->Npart;i++){
		int j = Node->ObjectList[i];
		for(int k=0;k<3;k++){
			triplet[k] = sign( R->Pos[3*j+k]-Node->Center[k] );
		}
		int oct   = get_octant(triplet); 

		if(NpartOctant[oct]==0){
			allocate_new_octree_node( Node, oct, label );
		}

		indexlist[oct][NpartOctant[oct]] = j;

		NpartOctant[oct]++;
		Node->Children[oct]->Npart++;


	}

	for(int oct=0;oct<8;oct++){

		if( NpartOctant[oct] == 0 ){
			continue;
		}

		Node->Children[oct]->ObjectList = calloc(Node->Children[oct]->Npart,sizeof(int));

		for(int i=0;i<Node->Children[oct]->Npart;i++){
			Node->Children[oct]->ObjectList[i] = indexlist[oct][i];
		}

                get_triplet_from_octant(oct,triplet);

		for(int k=0;k<3;k++){
			Node->Children[oct]->Center[k] = Node->Center[k] + triplet[k]*Node->Children[oct]->HalfSize;
		}

		for(int i=0;i<Node->Children[oct]->Npart;i++){
			ParticleInNode[ Node->Children[oct]->ObjectList[i] ] = Node->Children[oct];
		}

		if( Node->Children[oct]->Npart == 1 ){
			Node->Children[oct]->IsLeaf = 1;
		}

	}

	for(int oct=0;oct<8;oct++){
		if( Node->ChildExists[oct] ){
			if( Node->Children[oct]->Npart > MaxLeafSize ){
				octree( R, Node->Children[oct], MaxLeafSize, label );
			} else {
				Node->Children[oct]->IsLeaf = 1;
			}
		}
	}

}

void spatialpartition( struct RegularizedRegion *R, int MaxLeafSize ){

	double rmax = get_rmax( R );

	OctreeRootNode = calloc(1,sizeof(struct OctreeNode));

	OctreeRootNode->ID          = 0;
	OctreeRootNode->Parent      = NULL;
        OctreeRootNode->HalfSize    = 1.001*rmax;
	OctreeRootNode->Npart       = R->NumVertex;
	OctreeRootNode->ObjectList  = calloc(R->NumVertex,sizeof(int));
	OctreeRootNode->level       = 0;
	for(int i=0;i<3;i++){
		OctreeRootNode->Center[i] = 0.0;
	}
	for(int i=0;i<8;i++){
		OctreeRootNode->ChildExists[i] = 0;
	}

	for(int i=0;i<R->NumVertex;i++){
		OctreeRootNode->ObjectList[i] = i;
	}
	
	int label = 1;
	octree( R, OctreeRootNode, MaxLeafSize, &label );
}

int get_longest( double w[3], int id[3] ){

	double L=-1;
	int ind=-1;
	for(int m=0;m<3;m++){
		if(w[m]>L){
			L = w[m];
			ind = m;
		}
	}
	return id[ind];
}

int get_edge( int Nleafs, int va, int vb ){

	int N = Nleafs-2;
	int lo,hi;
	lo = va, hi = vb;
	if( va>vb ){
		lo = vb;
		hi = va;
	}
	int index=hi-1;
	for(int p=0;p<lo;p++){
		index+=N;
		N--;
	}
	return index;
}

void SolveMetaMST( struct RegularizedRegion *R, int Nleafs, int *Npart, int **Particles, int *NumEdge ){

	int Nedge = Nleafs*(Nleafs-1)/2;
	struct WeightIndex2 *W       = calloc(Nedge,sizeof(struct WeightIndex2));
        int *InMst = calloc(Nleafs,sizeof(int));

	int ThisBlock,CumNumEdge=0,istart=0,jstart;
#ifdef PARALLEL
        loop_scheduling_N2(Nleafs,Ntask,&istart,&jstart,&ThisBlock,&CumNumEdge,ThisTask);
#endif

	struct OctreeNode *Node_i, *Node_j;
        int counter = 0, ind1, ind2, loop=1;

	int    *ind1s   = calloc(Nedge,sizeof(int));
	int    *ind2s   = calloc(Nedge,sizeof(int));
	double *weights = calloc(Nedge,sizeof(double));

	counter = 0;
        for(int i=istart;i<Nleafs;i++){

		Node_i = ParticleInNode[ Particles[i][0] ];

#ifdef PARALLEL
                if(i>istart){
                        jstart = i+1;
                }
                for(int j=jstart;j<Nleafs;j++){
#else
                for(int j=i+1;j<Nleafs;j++){
#endif

			Node_j = ParticleInNode[ Particles[j][0] ];

			double r2 = 0;
			for(int k=0;k<3;k++){
				double ds = Node_i->Center[k]-Node_j->Center[k];
				r2 = ds*ds;
			}        
			double NodeSeparation = sqrt(r2);
			double NodeSize       = 1.01*sqrt(2)*(Node_i->HalfSize+Node_j->HalfSize);  // with safety buffer

			if( NodeSeparation <= NodeSize ){


	                	double MinWeight = 1e10;
                        	for(int q=0;q<Npart[i];q++){
                                	for(int p=0;p<Npart[j];p++){
                                        	double r2 = 0;
                                        	int i1 = Particles[i][q];
                                        	int i2 = Particles[j][p];
                                        	for(int k=0;k<3;k++){
                                                	double ds = R->Pos[3*i1+k]-R->Pos[3*i2+k];
                                                	r2 += ds*ds;
                                        	}
                                        	double r = sqrt(r2);
                                        	if( r<MinWeight ){
                                                	MinWeight = r;
                                                	ind1 = i1;
                                                	ind2 = i2;
                                        	}
                                	}
                        	}

			} else {

				double MinWeight = 1e10;
                                for(int q=0;q<Npart[i];q++){
					int i1 = Particles[i][q];
					double r2 = 0;
					for(int k=0;k<3;k++){
                                                double ds = R->Pos[3*i1+k]-Node_j->Center[k];
                                                r2 += ds*ds;
                                        }
                                        double r = sqrt(r2);
                                        if( r<MinWeight ){
                                        	MinWeight = r;
                                                ind1 = i1;
                                        }
				}
				MinWeight = 1e10;
                                for(int p=0;p<Npart[j];p++){
                                        int i2 = Particles[j][p];
                                        double r2 = 0;
                                        for(int k=0;k<3;k++){
                                                double ds = R->Pos[3*i2+k]-Node_i->Center[k];
                                                r2 += ds*ds;
                                        }
                                        double r = sqrt(r2);
                                        if( r<MinWeight ){
                                                MinWeight = r;
                                                ind2 = i2;
                                        }
                                }


			}

			r2 = 0;
                        for(int k=0;k<3;k++){
                        	double ds = R->Pos[3*ind1+k]- R->Pos[3*ind2+k];
                        	r2 += ds*ds;
                        }
                        double r = sqrt(r2);

                        weights[CumNumEdge+counter] = r;
                        ind1s[CumNumEdge+counter] = ind1;
                        ind2s[CumNumEdge+counter] = ind2;	
                        counter++;

#ifdef PARALLEL
                        if(counter == ThisBlock ){
                                loop=0;
                                break;
                        }
                }
                if(loop==0){
                        break;
                }
#else
                }
#endif
        }


#ifdef PARALLEL
	MPI_Allreduce(MPI_IN_PLACE,ind1s,Nedge,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE,ind2s,Nedge,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE,weights,Nedge,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
#endif

	counter = 0;
        for(int i=0;i<Nleafs;i++){
                for(int j=i+1;j<Nleafs;j++){
			W[counter].weight = weights[counter];
			W[counter].id1   = ind1s[counter];
			W[counter].id2   = ind2s[counter];
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
	int block     = (int) floor(Nedge/Ntask);
	istart        = ThisTask*block; 
	istop     = (ThisTask+1)*block;
	if(ThisTask==Ntask-1) istop = Nedge;
#endif

	int *flag = calloc(Nedge,sizeof(int));

	int Noff=0;
	double w[3];
	int  ind[3];
	for(int e1=istart;e1<istop;e1++){

		int v1 = W[e1].index1;
		int v2 = W[e1].index2;
		ind[0] = e1;
		w[0] = W[e1].weight;

		if(w[0]<0) continue;

		for(int v3=0;v3<Nleafs;v3++){

			if(w[0]<0) continue;

			if( v3==v1 || v3==v2 ) continue;

			ind[1] = get_edge( Nleafs, v1, v3 );
			ind[2] = get_edge( Nleafs, v2, v3 );

			w[1] = W[ind[1]].weight;
			w[2] = W[ind[2]].weight;
			if(w[1]<0||w[2]<0) continue;
			int elong = get_longest(w,ind);

			flag[elong] = 1;
			W[elong].weight *= -1;

			//int v3 = W[e2].index1;
                	//int v4 = W[e2].index2;
			//int v5 = W[e3].index1;
                        //int v6 = W[e3].index2;

			//if(ThisTask==0) printf("(%d,%d)  (%d,%d)  (%d,%d)\n",v1,v2,v3,v4,v5,v6);

		}

	}


#ifdef PARALLEL
	MPI_Allreduce(MPI_IN_PLACE,flag,Nedge,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
#endif

	for(int e1=0;e1<Nedge;e1++){
		if(flag[e1]==0) Noff++;
	}


	struct WeightIndex2 *Wfilter = calloc(Nedge-Noff,sizeof(struct WeightIndex2));


	counter = 0;
	for(int e1=0;e1<Nedge;e1++){
                if( flag[e1]==0 ){
			Wfilter[counter].weight = W[e1].weight;
			Wfilter[counter].id1    = W[e1].id1;
			Wfilter[counter].id2    = W[e1].id2;
			Wfilter[counter].index1 = W[e1].index1;
			Wfilter[counter].index2 = W[e1].index2;
			counter++;
		}
        }


	Nedge -= Noff;
	free(W);
	free(flag);



qsort(Wfilter,Nedge,sizeof(struct WeightIndex2),cmp_weight_index2);

InMst[0] = 1;

        int remaining = Nleafs-1;

        //FILE *fout;
        //char filename[100];
        //sprintf(filename,"output_mst/meta_mst.txt");
        //if(ThisTask==0) fout = fopen(filename,"w");

	istart = 0;
	istop = Nedge;
#ifdef PARALLEL
	block = (int) floor(Nedge/Ntask);
	istart = ThisTask*block;
	istop  = (ThisTask+1)*block;
	if(ThisTask==Ntask-1) istop = Nedge;
#endif

        do {


		struct WeightIndex wi;
		wi.weight = 1e10;
                wi.index  = -1;

        	for(int i=istart;i<istop;i++){
                	int i1 = Wfilter[i].index1;
                	int i2 = Wfilter[i].index2;
                	if( InMst[i1] != InMst[i2] ){
                       		wi.index = i;
				wi.weight = Wfilter[i].weight;
                       		break;
                	}
        	}

#ifdef PARALLEL
		MPI_Allreduce( MPI_IN_PLACE, &wi, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD );
#endif

		InMst[ Wfilter[ wi.index ].index1  ] = 1;
		InMst[ Wfilter[ wi.index ].index2  ] = 1;

		R->EdgeInMST[*NumEdge].vertex1 = Wfilter[ wi.index ].id1;
                R->EdgeInMST[*NumEdge].vertex2 = Wfilter[ wi.index ].id2;
                R->EdgeInMST[*NumEdge].weight  = Wfilter[ wi.index ].weight;
		(*NumEdge)++;

        	remaining--;
        } while( remaining>0 );


        //if(ThisTask==0) fflush(fout);
        //if(ThisTask==0) fclose(fout);

	free(Wfilter);
	free(InMst);

}

void SolveSubMST( struct RegularizedRegion *R, int Npart, int *Particles, int label, int *NumEdge ){

	if(Npart==1) return;
	//if(ThisTask==0) printf("---> SOLVE SUBMST: %d particles\n", Npart);

	int Nedge  = Npart*(Npart-1)/2;
	struct WeightIndex2 *W = calloc(Nedge,sizeof(struct WeightIndex2)); 
	int *InMst = calloc(Npart,sizeof(int));

	int counter = 0;
	for(int i=0;i<Npart;i++){
		int ind1 = Particles[i];
		for(int j=i+1;j<Npart;j++){
			int ind2 = Particles[j];
			double r2=0;
			for(int k=0;k<3;k++){
				double ds = R->Pos[3*ind1+k] - R->Pos[3*ind2+k];
				r2 += ds*ds;
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

	clock_t aika0 = clock();

	qsort(W,Nedge,sizeof(struct WeightIndex2),cmp_weight_index2);
	clock_t aika1 = clock();

	loopt4 += (double) (aika1-aika0)/CLOCKS_PER_SEC;

	InMst[0] = 1;

	int remaining = Npart-1;

	//FILE *fout;
	//char filename[100];
	//sprintf(filename,"output_mst/mst_%04d.txt",label);
	//if(ThisTask==0) fout = fopen(filename,"w");

	do {

	for(int i=0;i<Nedge;i++){
		int i1 = W[i].index1;
		int i2 = W[i].index2;
		if( InMst[i1] != InMst[i2] ){
			if( InMst[i1]==1 ){
				InMst[i2]=1;
			} else {
				InMst[i1]=1;
			}

			R->EdgeInMST[*NumEdge].vertex1 = W[i].id1;
	                R->EdgeInMST[*NumEdge].vertex2 = W[i].id2;
			R->EdgeInMST[*NumEdge].weight  = W[i].weight;
			(*NumEdge)++;

			//if(ThisTask==0) fprintf(fout, "%.20lf %.20lf %.20lf %.20lf %d %d\n", R->Pos[3*W[i].id1+0], R->Pos[3*W[i].id1+1], R->Pos[3*W[i].id2+0], R->Pos[3*W[i].id2+1], h1, h2);
			break;
		}
	}

	remaining--;
        } while( remaining>0 );

	//if(ThisTask==0) fflush(fout);
	//if(ThisTask==0) fclose(fout);

        free(W);
        free(InMst);
}

void SwapEdge( struct RegularizedRegion *R, int i, int j ){

	int v1   = R->EdgeInMST[i].vertex1;
	int v2   = R->EdgeInMST[i].vertex2;
	double w = R->EdgeInMST[i].weight;

	R->EdgeInMST[i].vertex1 = R->EdgeInMST[j].vertex1;
	R->EdgeInMST[i].vertex2 = R->EdgeInMST[j].vertex2;
	R->EdgeInMST[i].weight  = R->EdgeInMST[j].weight;

	R->EdgeInMST[j].vertex1 = v1;
	R->EdgeInMST[j].vertex2 = v2;
	R->EdgeInMST[j].weight  = w;

}

void CombineAllMST( struct RegularizedRegion *R, int NumEdge ){

	int next;
	double rmin=DBL_MAX;
	for(int i=0;i<R->NumVertex;i++){
		double r2=0;
                for(int k=0;k<3;k++){
                        double ds = R->Pos[3*i+k];
                        r2 += ds*ds;
                }
                if( rmin>sqrt(r2) ){
                        rmin = sqrt(r2);
                        next = i;
                }

        }

	int counter=0;
	R->Vertex[next].inMST = 1;

        int istart = 0;
        int istop = NumEdge;
#ifdef PARALLELBUG
        int block = (int) floor(NumEdge/Ntask);
        istart = ThisTask*block;
        istop  = (ThisTask+1)*block;
        if(ThisTask==Ntask-1) istop = NumEdge;
#endif

	double sumweight = 0;

	do {

		struct WeightIndex wi;
		wi.weight = 1e10;
		wi.index  = -1;

		for(int i=counter;i<istop;i++){
			int v1 = R->EdgeInMST[i].vertex1;
			int v2 = R->EdgeInMST[i].vertex2;
			if( R->Vertex[v1].inMST != R->Vertex[v2].inMST ){
				if( R->EdgeInMST[i].weight<wi.weight ){
					wi.weight = R->EdgeInMST[i].weight;
					wi.index  = i;
				}
			}
		}

#ifdef PARALLELBUG
                MPI_Allreduce( MPI_IN_PLACE, &wi, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD );
#endif

		sumweight += wi.weight;

		int new = R->EdgeInMST[wi.index].vertex1;
		int old = R->EdgeInMST[wi.index].vertex2;
		if( R->Vertex[new].inMST == 1 ){
			old = R->EdgeInMST[wi.index].vertex1;
			new = R->EdgeInMST[wi.index].vertex2;
		}

		R->Vertex[new].inMST = 1;
                R->Vertex[new].parent = old;
                R->Vertex[new].level  = R->Vertex[old].level+1;
                R->Vertex[new].edgetoparent = counter;

                for(int k=0;k<3;k++){
 	               R->State[3*counter+k]                    = R->Pos[3*new+k] - R->Pos[3*old+k];
                       R->State[3*(R->NumVertex-1)+3*counter+k] = R->Vel[3*new+k] - R->Vel[3*old+k];
                }

		SwapEdge( R, counter, wi.index );
		counter++;

	} while(counter<NumEdge);


	if(ThisTask==0){
		printf("%.10e\n",sumweight);
		fflush(stdout);
	}



#if 0
	for(int i=0;i<NumEdge;i++){
		int v1 = R->EdgeInMST[i].vertex1;
		int v2 = R->EdgeInMST[i].vertex2;
		if(ThisTask==0){
			printf("%.20lf %.20lf %.20lf %.20lf\n", R->Pos[3*v1+0], R->Pos[3*v1+1], R->Pos[3*v2+0], R->Pos[3*v2+1]);
		}
	}
#endif

}


void DivideAndConquerPrim( struct RegularizedRegion *R ){

	my_barrier();
	clock_t t0 = clock();


        for(int i=0;i<R->NumVertex;i++){
                R->Vertex[i].id = i;
                R->Vertex[i].degree = 0;
                R->Vertex[i].edgetoparent = -1;
                R->Vertex[i].parent = -1;
                R->Vertex[i].level  = 0;
                R->Vertex[i].inMST  = 0;
        }

	int Nleafs = 0;
	int MaxLeafSize = (int) floor( 3*sqrt(R->NumVertex) );

	clock_t t10 = clock();
        spatialpartition( &R[0], MaxLeafSize );


	GetNumberOfLeafs( OctreeRootNode, &Nleafs );
	int *LeafSizes = calloc(Nleafs,sizeof(int));
	int **LeafParticles = calloc(Nleafs,sizeof(int *));
	clock_t t11 = clock();
        tprim1 += (double) (t11-t10)/CLOCKS_PER_SEC;


	clock_t t20 = clock();
	Nleafs = 0;
	list_octree_level(OctreeRootNode, LeafSizes, LeafParticles, &Nleafs );
	clock_t t21 = clock();
        tprim2 += (double) (t21-t20)/CLOCKS_PER_SEC;


	clock_t t30 = clock();
	int NumEdge = 0;
	for(int i=0;i<Nleafs;i++){
		SolveSubMST( R, LeafSizes[i], LeafParticles[i], i, &NumEdge );
	}
	clock_t t31 = clock();
        tprim3 += (double) (t31-t30)/CLOCKS_PER_SEC;


	clock_t t40 = clock();
	SolveMetaMST(R, Nleafs, LeafSizes, LeafParticles, &NumEdge );
	clock_t t41 = clock();
        tprim4 += (double) (t41-t40)/CLOCKS_PER_SEC;


	clock_t t50 = clock();
	CombineAllMST( R, NumEdge );
	clock_t t51 = clock();
	tprim5 += (double) (t51-t50)/CLOCKS_PER_SEC;

	free_octree(OctreeRootNode);
	for(int i=0;i<Nleafs;i++){
		free(LeafParticles[i]);
	}
	free(LeafParticles);
	free(LeafSizes);

	clock_t t1 = clock();
	time_mst += (double) (t1-t0)/CLOCKS_PER_SEC;

}


int EdgeSearchForPrimSlow( struct RegularizedRegion *R, int NumLocalEdges, double *w ){

		clock_t mikkolatime = clock();
		struct WeightIndex wi;

                wi.weight = 1e10;
                wi.index  = -1;
                for(int i=0;i<NumLocalEdges;i++){
                        if( R->Vertex[R->LocalEdge[i].vertex1].inMST != R->Vertex[R->LocalEdge[i].vertex2].inMST ){
                                wi.weight = R->LocalEdge[i].weight;
                                wi.index  = R->LocalEdge[i].id;
                                break;
                        }
                }


                clock_t sverretime = clock();
                tprim3 += (double) (sverretime-mikkolatime)/CLOCKS_PER_SEC;

                mikkolatime = clock();
#ifdef PARALLEL
                MPI_Allreduce( MPI_IN_PLACE, &wi, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD );
#endif

		*w = wi.weight;

                sverretime = clock();
                tprim4 += (double) (sverretime-mikkolatime)/CLOCKS_PER_SEC;

		return wi.index;

}

int PresentInArray( int p, int *Array, int N ){

	int result = -1;
	for(int i=0;i<N;i++){
		if( Array[i] == p ){
			result = i;
			break;
		}
	}
	return result;
}

int get_heaviest( double a[3], int b1,int b2,int b3 ){

	int index = -1;
	double w  = -1;
	for(int i=0;i<3;i++){
		if(a[i]>w){
			w = a[i];
			index = i;
		}
	} 
	if(index==0) return b1;
	if(index==1) return b2;
	return b3;
}

void PrimMST(struct RegularizedRegion *R){

	msttime0 = clock();
	clock_t mytime0  = clock();

	int ThisBlock,CumNumEdge=0,istart=0,jstart;
#ifdef PARALLEL
	loop_scheduling_N2(R->NumVertex,Ntask,&istart,&jstart,&ThisBlock,&CumNumEdge,ThisTask);
#endif
	int NumLocalEdges=0; int loop=1; int c=0,counter=0;

	double rmin=1e10,r2,rmax=0;
	int last = -1;

	        for(int i=0;i<R->NumVertex;i++){
                r2 = 0;
                for(int k=0;k<3;k++){
                        double ds = R->Pos[3*i+k];
                        r2 += ds*ds;
                }
                if( rmin > sqrt(r2) ){
                        rmin = sqrt(r2);
                        last = i;
                }
        }

	for(int i=istart;i<R->NumVertex;i++){

#ifdef PARALLEL
                if(i>istart){
                        jstart = i+1;
                }
                for(int j=jstart;j<R->NumVertex;j++){
#else
                for(int j=i+1;j<R->NumVertex;j++){
#endif
			counter++;
			double w=0;
			for(int k=0;k<3;k++){
				double ds = R->Pos[3*i+k] - R->Pos[3*j+k];
				w += ds*ds;
			}
			w = sqrt(w);

			R->LocalEdge[NumLocalEdges].weight  = w;
			R->LocalEdge[NumLocalEdges].vertex1 = i;
			R->LocalEdge[NumLocalEdges].vertex2 = j;
			R->LocalEdge[NumLocalEdges].id      = (int) (CumNumEdge + c);

			NumLocalEdges++;
			c++;
			
#ifdef PARALLEL
			if(counter == ThisBlock ){
				loop=0;
				break;
			}
		}
		if(loop==0){
			break;
		}
#else
		}
#endif
	}


	qsort(R->LocalEdge,NumLocalEdges,sizeof(struct GraphEdge),cmp);

	int remaining = 0;

	struct WeightIndex wi;

	for(int i=0;i<R->NumVertex;i++){
		R->Vertex[i].id = i;
		R->Vertex[i].degree = 0;
		R->Vertex[i].edgetoparent = -1;
		R->Vertex[i].parent = -1;
		R->Vertex[i].level  = 0;
		R->Vertex[i].inMST  = 0;
	}

        R->Vertex[last].inMST = 1;

	double sumweight = 0;

	c=0;
	do {

		double myw;
		int NextEdge = EdgeSearchForPrimSlow( R, NumLocalEdges, &myw );

		sumweight += myw;

		int v1,v2;	
		get_v1_v2( R->NumVertex, NextEdge, &v1,&v2 );

		R->EdgeInMST[c].vertex1 = v1;
                R->EdgeInMST[c].vertex2 = v2;

		R->Vertex[v1].degree++;
		R->Vertex[v2].degree++;

		if( R->Vertex[v1].inMST == 0  ){

			R->Vertex[v1].parent = v2;
			R->Vertex[v1].level  = R->Vertex[v2].level+1;
			R->Vertex[v1].inMST  = 1;
			R->Vertex[v1].edgetoparent = c;
			for(int k=0;k<3;k++){
				R->State[3*c+k] 			  = R->Pos[3*v1+k] - R->Pos[3*v2+k];
                                R->State[3*(R->NumVertex-1)+3*c+k] 	  = R->Vel[3*v1+k] - R->Vel[3*v2+k];
			}
		} else {

			R->Vertex[v2].parent = v1;
                        R->Vertex[v2].level  = R->Vertex[v1].level+1;
			R->Vertex[v2].inMST  = 1;
			R->Vertex[v2].edgetoparent = c;
			for(int k=0;k<3;k++){
				R->State[3*c+k] 			  = R->Pos[3*v2+k] - R->Pos[3*v1+k];
                                R->State[3*(R->NumVertex-1)+3*c+k] 	  = R->Vel[3*v2+k] - R->Vel[3*v1+k];
                        }
		}
		c++;
		remaining++;
	} while( remaining<R->NumVertex-1 );

	if(ThisTask==0) printf("%.10e\n",sumweight);

	msttime1 = clock();
        time_mst += (double) (msttime1-msttime0)/CLOCKS_PER_SEC;

}




void compute_U(struct RegularizedRegion *R){


	measure_cpu_time( CPU_TIME_MISC );

	clock_t thistime0 = clock();

	R->U    = 0;
	int ThisBlock,CumNumEdge,istart=0,jstart;
#ifdef PARALLEL
        loop_scheduling_N2(R->NumVertex,NumTaskPerGbsGroup,&istart,&jstart,&ThisBlock,&CumNumEdge,ThisTask_in_GbsGroup);
        int loop=1;
	int c=0;
#endif

	const int Nd = GLOBAL_ND;
	int d;
	int path[2];
	int sign[2];

        for(int i=istart;i<R->NumVertex;i++){

		double mi = R->Mass[i];
		int    Li = R->Vertex[i].level;
#ifdef PARALLEL
                if(i>istart){
                        jstart = i+1;
                }
                for(int j=jstart;j<R->NumVertex;j++){
#else
		for(int j=i+1;j<R->NumVertex;j++){
#endif
			double mj = R->Mass[j];
                	int    Lj = R->Vertex[j].level;

			double r2=0;

			int proximity = 0;

                        if( fabs(Li-Lj) < Nd ){
                                proximity = check_relative_proximity(j,i,Nd,R,&d,path,sign);
                        }

                        if(proximity==1 ){
				for(int k=0;k<3;k++){
					double dr = 0;
                                	for(int l=0;l<d;l++){
                                        	int index = path[l];
                                            	dr += sign[l]*R->State[3*index+k];
                                        }
					r2 += dr*dr;
                                }
                        } else {
                                for(int k=0;k<3;k++){
                                        double dr  = R->Pos[3*j+k] - R->Pos[3*i+k];
                                        r2 += dr*dr;
                                }
                        }

			const double invr = 1.0/sqrt(r2);
                        R->U += mi*mj*invr;
#ifdef PARALLEL
			c++;
			if(c == ThisBlock ){
                                loop=0;
                                break;
                        }
                }
                if(loop==0) break;
#else
		}
#endif

	}

	clock_t ctime0 = clock();
#ifdef PARALLEL
	if( NumTaskPerGbsGroup > 1 ){
		MPI_Allreduce(MPI_IN_PLACE,&R->U,   1,MPI_DOUBLE,MPI_SUM,MPI_COMM_GBS_GROUP);
	}
#endif
	clock_t ctime1 = clock();	
	time_comm += (double) (ctime1-ctime0)/CLOCKS_PER_SEC;

	if(!isfinite(R->U)){
		printf("U not finite.\n");
		exit(0);
	}

	R->U *= GCONST;

	clock_t thistime1 = clock();
        time_force += (double) (thistime1-thistime0)/CLOCKS_PER_SEC;

	measure_cpu_time( CPU_TIME_POTENTIALENERGY );

}

double estimate_timestep( struct RegularizedRegion *R ){

	double min_dt = DBL_MAX;
	for(int j=0;j<R->NumVertex;j++){
		double dt = sqrt( R->State[j]/R->MSTedgeAcc[j] );
		if( dt < min_dt  ){
			min_dt = dt;
		}
	}
	return min_dt;
}

void compute_U_and_Newtonian_Acc(struct RegularizedRegion *R){

	clock_t thistime0 = clock();

	measure_cpu_time( CPU_TIME_MISC );

	int ThisBlock,CumNumEdge,istart,jstart;
	int istop = R->NumVertex, jstop = R->NumVertex;

#ifdef PARALLEL
        loop_scheduling_N2(R->NumVertex,NumTaskPerGbsGroup,&istart,&jstart,&ThisBlock,&CumNumEdge,ThisTask_in_GbsGroup);
        int loop=1;
        int c=0;
#else
	istart = 0;
	jstart = 0;
#endif
	const int Nd = GLOBAL_ND;
        int d;
        int path[Nd];
        int sign[Nd];

	for(int i=0;i<3*R->NumVertex;i++){
                R->Acc[i] = 0;
        }

        R->U    = 0;
	mydouble dr[3], r2;

	double *Pos = R->Pos;	
	double *Mass = R->Mass;
	struct GraphVertex *Vertex = R->Vertex;

	for(int i=istart;i<istop;i++){
		const double mi = Mass[i];
		const int    Li = Vertex[i].level;
#ifdef PARALLEL
		if(i>istart) jstart = i+1;
		for(int j=jstart;j<jstop;j++){
#else
		for(int j=i+1;j<jstop;j++){
#endif
			r2 = 0;
			const double mj = Mass[j];		
			const int    Lj = Vertex[j].level;

			int proximity = 0;
                        if( abs(Li-Lj) <= Nd ){
                                proximity = check_relative_proximity_ND_2(j,i,R,&d,path,sign);
                        }

                        if(proximity){
                                for(int k=0;k<3;k++){
					dr[k] = 0;
                                        for(int l=0;l<d;l++){
                                                dr[k] += sign[l]*R->State[3*path[l]+k];
                                        }
                                        r2    += dr[k]*dr[k];
                                }
                        } else {
				for(int k=0;k<3;k++){
					dr[k] = Pos[3*j+k] - Pos[3*i+k];
					r2 += dr[k]*dr[k];					
				}
			}

			const mydouble invr = 1.0/sqrt(r2);
			R->U += mi*mj*invr;

			const mydouble invr3 = invr*invr*invr;

               	        for(int k=0;k<3;k++){
				double drinvr3 = dr[k]*invr3;
       	                        R->Acc[3*i+k] += mj*drinvr3;;
				R->Acc[3*j+k] += -mi*drinvr3;;
                        }
#ifdef PARALLEL
			c++;
                        if(c == ThisBlock ){
                                loop=0;
                                break;
                        }
#endif
		}
#ifdef PARALLEL
		if(loop==0) break;
#endif
	}

	clock_t ctime0 = clock();
#ifdef PARALLEL
        if( NumTaskPerGbsGroup > 1 ){
                MPI_Allreduce(MPI_IN_PLACE,R->Acc,  3*R->NumVertex,MPI_DOUBLE,MPI_SUM,MPI_COMM_GBS_GROUP);
                MPI_Allreduce(MPI_IN_PLACE,&R->U,   1,             MPI_DOUBLE,MPI_SUM,MPI_COMM_GBS_GROUP);
        }
#endif
	clock_t ctime1 = clock();
        time_comm += (double) (ctime1-ctime0)/CLOCKS_PER_SEC;


	int hi,lo;
        for(int i=0;i<R->NumVertex-1;i++){

                int v1,v2;
		get_v1_v2_fast( R, i, &v1,&v2 );

                if( R->Vertex[v1].level > R->Vertex[v2].level ){
                        hi = v1;
                        lo = v2;
                } else {
                        hi = v2;
                        lo = v1;
                }
                for(int k=0;k<3;k++){
                        R->MSTedgeAcc[3*i+k] = GCONST*(R->Acc[3*hi+k] - R->Acc[3*lo+k]);
                }
        }

	R->U *= GCONST;

	clock_t thistime1 = clock();
	time_force += (double) (thistime1-thistime0)/CLOCKS_PER_SEC;

	measure_cpu_time( CPU_TIME_ACC );
}


void into_Cartesian_from_MST(struct RegularizedRegion *R){

	clock_t timing0 = clock();

	int v1,v2;
	get_v1_v2_fast( R, 0, &v1,&v2 );

	int first;
	if(R->Vertex[v1].level == 0){
		first = v1;
	} else {
		first = v2;
	}
	for(int k=0;k<3;k++){
		R->Pos[3*first+k] = 0.0;
		R->Vel[3*first+k] = 0.0;
	}
	for(int i=0;i<R->NumVertex-1;i++){

		get_v1_v2_fast( R, i, &v1,&v2 );

		int sign,lo,hi;
		if( R->Vertex[v1].level > R->Vertex[v2].level ){
			lo   = v2;
			hi   = v1;
		} else {
			lo   = v1;
			hi   = v2;
		}
		for(int k=0;k<3;k++){
                	R->Pos[3*hi+k] = R->Pos[3*lo+k] + R->State[3*i+k];
                	R->Vel[3*hi+k] = R->Vel[3*lo+k] + R->State[3*(R->NumVertex-1)+3*i+k];
        	}
	}

	double M=0,*CoM = calloc(3,sizeof(double)), *CoV = calloc(3,sizeof(double));	total_memory += 6*sizeof(double);

        for(int i=0;i<R->NumVertex;i++){
                M += R->Mass[i];
                for(int k=0;k<3;k++){
                        CoM[k] += R->Mass[i]*R->Pos[3*i+k];
                        CoV[k] += R->Mass[i]*R->Vel[3*i+k];
                }
        }
        for(int k=0;k<3;k++){
                CoM[k] /= M;
                CoV[k] /= M;
        }

        for(int i=0;i<R->NumVertex;i++){
                for(int k=0;k<3;k++){
                        R->Pos[3*i+k] -= CoM[k];
                        R->Vel[3*i+k] -= CoV[k];
                }
        }

	free(CoM);
	free(CoV);		total_memory -= 6*sizeof(double);

	clock_t timing1 = clock();
        time_coord1 += (double) (timing1-timing0)/CLOCKS_PER_SEC;
}

void into_CoM_frame(struct RegularizedRegion *R){

	clock_t timing0 = clock();

	for(int k=0;k<3;k++){
		R->CoM_Pos[k] = 0;
		R->CoM_Vel[k] = 0;
	}
	double M=0;
	for(int i=0;i<R->NumVertex;i++){
		M += R->Mass[i];
		for(int k=0;k<3;k++){
                	R->CoM_Pos[k] += R->Mass[i]*R->Pos[3*i+k];
                	R->CoM_Vel[k] += R->Mass[i]*R->Vel[3*i+k];
        	}
	}
	for(int k=0;k<3;k++){
                R->CoM_Pos[k] /= M;
        	R->CoM_Vel[k] /= M;
        }
	for(int i=0;i<R->NumVertex;i++){
		for(int k=0;k<3;k++){
			R->Pos[3*i+k] -= R->CoM_Pos[k];
			R->Vel[3*i+k] -= R->CoM_Vel[k];
		}
	}

	clock_t timing1 = clock();
        time_coord2 += (double) (timing1-timing0)/CLOCKS_PER_SEC;
}

void compute_T(struct RegularizedRegion *R){
	R->T=0;
	for(int i=0;i<R->NumVertex;i++){
		double dv,v2 = 0;
		for(int k=0;k<3;k++){
			dv  = R->Vel[3*i+k];
			v2 += dv*dv;
		}
		R->T += 0.5*R->Mass[i]*v2;
	}

}

void update_pos(struct RegularizedRegion *R){

	clock_t timing0 = clock();

	int v1,v2;
	get_v1_v2_fast( R, 0, &v1,&v2 );

        int first;
        if(R->Vertex[v1].level == 0){
                first = v1;
        } else {
                first = v2;
        }
        for(int k=0;k<3;k++){
                R->Pos[3*first+k] = 0.0;
        }
        for(int i=0;i<R->NumVertex-1;i++){

		get_v1_v2_fast( R, i, &v1,&v2 );

                int sign,lo,hi;
                if( R->Vertex[v1].level > R->Vertex[v2].level ){
                        lo   = v2;
                        hi   = v1;
                } else {
                        lo   = v1;
                        hi   = v2;
                }
                for(int k=0;k<3;k++){
                        R->Pos[3*hi+k] = R->Pos[3*lo+k] + R->State[3*i+k];
                }
        }

	double M=0,*CoM = calloc(3,sizeof(double));	total_memory += 3*sizeof(double);
        for(int i=0;i<R->NumVertex;i++){
                M += R->Mass[i];
                for(int k=0;k<3;k++){
                        CoM[k] += R->Mass[i]*R->Pos[3*i+k];
                }
        }
        for(int k=0;k<3;k++){
                CoM[k] /= M;
        }
        for(int i=0;i<R->NumVertex;i++){
                for(int k=0;k<3;k++){
                        R->Pos[3*i+k] -= CoM[k];
                }
        }
	free(CoM);		total_memory -= 3*sizeof(double);

	clock_t timing1 = clock();
	time_coord3 += (double) (timing1-timing0)/CLOCKS_PER_SEC;
}

void update_vel(struct RegularizedRegion *R, double *Vel, enum velocity_type propagated_velocity){

	clock_t timing0 = clock();

	int v1,v2;
	get_v1_v2_fast( R, 0, &v1,&v2 );

        int first;
        if(R->Vertex[v1].level == 0){
                first = v1;
        } else {
                first = v2;
        }
        for(int k=0;k<3;k++){
                Vel[3*first+k] = 0.0;
        }
        for(int i=0;i<R->NumVertex-1;i++){

		get_v1_v2_fast( R, i, &v1,&v2 );

                int sign,lo,hi;
                if( R->Vertex[v1].level > R->Vertex[v2].level ){
                        lo   = v2;
                        hi   = v1;
                } else {
                        lo   = v1;
                        hi   = v2;
                }
                for(int k=0;k<3;k++){
if( propagated_velocity == PHYSICAL ){
                        Vel[3*hi+k] = Vel[3*lo+k] + R->State[3*(R->NumVertex-1)+3*i+k];
} else {
			Vel[3*hi+k] = Vel[3*lo+k] + R->AuxEdgeVel[3*i+k];
}
                }
        }

        double M=0,*CoV = calloc(3,sizeof(double));		total_memory += 3*sizeof(double);
        for(int i=0;i<R->NumVertex;i++){
                M += R->Mass[i];
                for(int k=0;k<3;k++){
                        CoV[k] += R->Mass[i]*Vel[3*i+k];
                }
        }
        for(int k=0;k<3;k++){
                CoV[k] /= M;
        }
        for(int i=0;i<R->NumVertex;i++){
                for(int k=0;k<3;k++){
                        Vel[3*i+k] -= CoV[k];
                }
        }
	free(CoV);		total_memory -= 3*sizeof(double);

	clock_t timing1 = clock();
	time_coord4 += (double) (timing1-timing0)/CLOCKS_PER_SEC;

}

void drift( double ds, struct RegularizedRegion *R ){

	compute_T(R);
	double dt = ds/(R->T+R->State[R->BIndex]);
	R->State[R->TimeIndex] += dt;

	for(int i=0;i<R->NumVertex-1;i++){
		for(int k=0;k<3;k++){
			R->State[3*i+k] += dt*R->State[3*(R->NumVertex-1)+3*i+k];
		}
	}
	update_pos(R);

}

void set_auxiliary_variables( struct RegularizedRegion *R ){

	for(int i=0;i<3*R->NumVertex;i++){
		R->AuxVel[i] = R->Vel[i];
	}
	int offset = 3*(R->NumVertex-1);
	for(int i=0;i<3*(R->NumVertex-1);i++){
                R->AuxEdgeVel[i] = R->State[offset+i];
        }

}

void physical_kick( double dt, struct RegularizedRegion *R ){

	int offset = 3*(R->NumVertex-1);
	update_vel(R, R->AuxVel, AUXILIARY);
        compute_Post_Newtonian_Acc(R, R->AuxVel);

	double dB = 0;
	for(int i=0;i<R->NumVertex;i++){
		for(int k=0;k<3;k++){
			dB += R->Mass[i]*R->AuxVel[3*i+k]*R->AccPN[3*i+k];
		}
	}
	R->State[R->BIndex] -= dB*dt;

        for(int i=0;i<R->NumVertex-1;i++){
                for(int k=0;k<3;k++){
                        R->State[offset+3*i+k] += dt * ( R->MSTedgeAcc[3*i+k] + R->MSTedgeAcc_PN[3*i+k] );
                }
        }
}

void auxiliary_kick( double dt, struct RegularizedRegion *R ){

	update_vel(R, R->Vel, PHYSICAL);
	compute_Post_Newtonian_Acc(R, R->Vel);

	for(int i=0;i<R->NumVertex-1;i++){
                for(int k=0;k<3;k++){
                        R->AuxEdgeVel[3*i+k] += dt * ( R->MSTedgeAcc[3*i+k] + R->MSTedgeAcc_PN[3*i+k] );
                }
        }
}


void kick( double ds, struct RegularizedRegion *R ){

	compute_U_and_Newtonian_Acc(R);
	double dt = ds/R->U;

#if 1
	for(int i=0;i<R->NumVertex-1;i++){
                for(int k=0;k<3;k++){
                        R->State[3*(R->NumVertex-1)+3*i+k] += dt * R->MSTedgeAcc[3*i+k];
                }
        }
#else
	auxiliary_kick( dt/2, R );
	physical_kick(  dt,   R );
        auxiliary_kick( dt/2, R );
#endif
	update_vel(R, R->Vel, PHYSICAL);

}

double get_n_double(int i){
	return (2.0*(i+1.0));
}
int get_n_int(int i){
        return (2.0*(i+1.0));
}

double get_error( double a, double b ){
	return fabs(b-a);
}

int gbs_group_with_min_workload( int *GbsGroupWorkLoad ){
	int min_workload = INT_MAX;
	int index        = 0;
	for(int i=0;i<NumGbsGroup;i++){
		if( GbsGroupWorkLoad[i]<min_workload ){
			min_workload = GbsGroupWorkLoad[i];
			index = i;
		}
	}
	return index;
}

void divide_computational_load( struct RegularizedRegion *R, int NumberOfSystems ){

	int *GbsGroupWorkLoad = calloc(NumGbsGroup,sizeof(int));	total_memory += NumGbsGroup*sizeof(int);

	int counter = 0;
	for(int region=0;region<NumberOfSystems;region++){
		int N = R[region].NumVertex;
		for(int i=KMAX-1;i>=0;i--){
			int ThisSubdivisionCost 				    = get_n_int(i);	// *N*(N-1)/2;
			int GroupIndex 						    = gbs_group_with_min_workload(GbsGroupWorkLoad);
			GbsGroupWorkLoad[GroupIndex] 				   += ThisSubdivisionCost;
			ComputationToDoList.ComputationalTask[i].WhichRegion 	    = region;
			ComputationToDoList.ComputationalTask[i].GroupToPerformTask = GroupIndex;
			ComputationToDoList.ComputationalTask[i].ThisR 	            = &R[region];
			ComputationToDoList.ComputationalTask[i].NumberOfSubsteps   = get_n_int(i);
			counter++;
		}
	}
	ComputationToDoList.NumberOfComputationalTasks = counter;

	free(GbsGroupWorkLoad);		total_memory -= NumGbsGroup*sizeof(int);

#if 0
	for(int i=0;i<NumberOfSystems*KMAX;i++){
		if(ThisTask==0){
			printf("system %d responsible group %d substeps %d\n",ComputationToDoList.ComputationalTask[i].WhichRegion,ComputationToDoList.ComputationalTask[i].GroupToPerformTask, ComputationToDoList.ComputationalTask[i].NumberOfSubsteps );
			fflush(stdout);
		}
	}
	exit(0);
#endif

}


void compute_total_energy( struct RegularizedRegion *R ){

        compute_T(R);
        compute_U(R);
        R->H = R->T - R->U;
        R->B = -R->H;
	R->State[R->BIndex] = R->B;
}


void mst_leapfrog( struct RegularizedRegion *R, double Hstep, int NumberOfSubsteps, int *flag ){

	double h = Hstep/NumberOfSubsteps;
	set_auxiliary_variables(R);

        drift(h/2,R);
        for(int step=0;step<NumberOfSubsteps-1;step++){
                kick(h,R);
                drift(h,R);
        }
        kick(h,R);
        drift(h/2,R);

#if 0

	if(NumberOfSubsteps>2) return;

	*flag = 0;

	compute_U( R );
	double U = R->U;
	double T = R->T;
	double B = R->State[ R->BIndex ];

	int index = (NumberOfSubsteps/2)-1;
	ttlratio[index] = fabs( U/(T+B) - 1.0 );

	if( ttlratio[index] > 1e-3 ){
		*flag = 1;
	}

#endif

}

void copy_regularized_region( struct RegularizedRegion *Orig, struct RegularizedRegion *Copy ){

	Copy->gbs_tolerance	    = Orig->gbs_tolerance;
        Copy->output_time_tolerance = Orig->output_time_tolerance;
        Copy->NumVertex       	    = Orig->NumVertex;
        Copy->NumEdge         	    = Orig->NumEdge;
        Copy->NumDynVariables 	    = Orig->NumDynVariables;
        Copy->TimeIndex       	    = Orig->TimeIndex;
        Copy->BIndex          	    = Orig->BIndex;
	Copy->Hstep                 = Orig->Hstep;

	for(int j=0;j<Orig->NumVertex;j++){
		Copy->Vertex[j].id 	= Orig->Vertex[j].id;
		Copy->Vertex[j].degree  = Orig->Vertex[j].degree;
		Copy->Vertex[j].type    = Orig->Vertex[j].type;
		Copy->Vertex[j].parent  = Orig->Vertex[j].parent;
		Copy->Vertex[j].edgetoparent = Orig->Vertex[j].edgetoparent;
		Copy->Vertex[j].level   = Orig->Vertex[j].level;
		Copy->Vertex[j].inMST   = Orig->Vertex[j].inMST;
	}
        for(int j=0;j<Orig->NumVertex-1;j++){
		Copy->EdgeInMST[j].vertex1 = Orig->EdgeInMST[j].vertex1;
		Copy->EdgeInMST[j].vertex2 = Orig->EdgeInMST[j].vertex2;
        }

	for(int j=0;j<Orig->NumDynVariables;j++){
		Copy->State[j] = Orig->State[j];
        }
        for(int j=0;j<3*Orig->NumVertex;j++){
		Copy->Pos[j]    = Orig->Pos[j];
		Copy->Vel[j]    = Orig->Vel[j];
		Copy->Acc[j]    = Orig->Acc[j];
		Copy->AuxVel[j] = Orig->AuxVel[j];
		Copy->AccPN[j]  = Orig->AccPN[j];
        }
	for(int j=0;j<3*(Orig->NumVertex-1);j++){
		Copy->MSTedgeAcc[j]    = Orig->MSTedgeAcc[j];
		Copy->AuxEdgeVel[j]    = Orig->AuxEdgeVel[j];
		Copy->MSTedgeAcc_PN[j] = Orig->MSTedgeAcc_PN[j];
        }
	for(int j=0;j<Orig->NumVertex;j++){
		Copy->Mass[j]  = Orig->Mass[j];
	}
}

void extrapolate_single_variable( double *y, const int N, double *result, double *error, double *yscal ){

        for(int i=0;i<N;i++){
                gbsS[i][0] = y[i];
        }
        for(int j=1;j<N;j++){
                for(int i=0;i<N-j;i++){
                        double nterm = pow(get_n_double(i+j)/get_n_double(i),2);
                        gbsS[i][j] = ( nterm*gbsS[i+1][j-1] - gbsS[i][j-1] ) / (nterm-1.0);
                }
        }
        for(int i=1;i<N;i++){
                result[i-1] = gbsS[0][i];
                error[i-1]  = fabs(gbsS[0][i]-gbsS[0][i-1]) / yscal[i];
        }

}

void get_scaling( int N, double *Mass, double *R, double *V, double T, double U ){

	double M = 0;

	for(int i=0;i<N;i++){
		M += Mass[i];
	}
	*V = sqrt(2*T/M);
	*R = fabs(GCONST*M*M/(2*U));


	if(ThisTask==0) printf("%e %e crossing time %.10e --- %e %e\n", 2*(*R)/(*V), *V, *R, T, U);

}

int extrapolate_all_variables( int NumberOfSystems, int korder ){

	clock_t aika0 = clock();

	int status;
	measure_cpu_time( CPU_TIME_MISC );

	for(int region=0;region<NumberOfSystems;region++){

		struct RegularizedRegion *ThisR;

		int *SubStepDivision = calloc(korder,sizeof(int));	total_memory += korder*sizeof(int);

		int counter = 0;
		int NumDynVariables;
		for(int i=0;i<NumberOfSystems*korder;i++){
			if( region == ComputationToDoList.ComputationalTask[i].WhichRegion ){
				ThisR = ComputationToDoList.ComputationalTask[i].ThisR;
				NumDynVariables = ThisR->NumDynVariables;
				SubStepDivision[counter] = i;
				counter++;
			}
		}
#if 0
                double *errorbuf  = calloc(NumDynVariables,sizeof(double));
                double *resultbuf = calloc(NumDynVariables,sizeof(double));

		double *y = calloc(korder,sizeof(double));		total_memory += korder*sizeof(double);
		double *yscal = calloc(korder,sizeof(double));		total_memory += korder*sizeof(double);
		double *maxerror = calloc(korder-1,sizeof(double));	total_memory += (korder-1)*sizeof(double);

		double **error,**result;
		error = calloc(NumDynVariables,sizeof(double *));	total_memory += NumDynVariables*sizeof(double *);
		result = calloc(NumDynVariables,sizeof(double *));	total_memory += NumDynVariables*sizeof(double *);
		for(int i=0;i<NumDynVariables;i++){
			error[i] = calloc(korder-1,sizeof(double));	total_memory += (korder-1)*sizeof(double);
			result[i] = calloc(korder-1,sizeof(double));	total_memory += (korder-1)*sizeof(double);
		}
#endif

		int block      = (int) floor(NumDynVariables/Ntask); 
		int loop_start = block*ThisTask;
		int loop_end   = block*(ThisTask+1);
		if(ThisTask==Ntask-1){
			loop_end = NumDynVariables;
		}

		clock_t mytime = clock();
        	text1 += (double) (mytime-aika0)/CLOCKS_PER_SEC;
        	aika0 =  clock();

		//loop_start = 0;
		//loop_end = NumDynVariables;


		for(int v=loop_start;v<loop_end;v++){
			for(int i=0;i<korder;i++){
				y[i]     =  ComputationToDoList.CopyOfSingleRegion[ SubStepDivision[i] ].State[v];
				yscal[i] = fabs(y[i]);
			}
			extrapolate_single_variable( y, korder, result[v], error[v], yscal );
		}


                mytime = clock();
                text2 += (double) (mytime-aika0)/CLOCKS_PER_SEC;
                aika0 =  clock();

		for(int i=0;i<NumDynVariables;i++){
                        errorbuf[i]  = 0;
                        resultbuf[i] = 0;
                }

		for(int i=loop_start;i<loop_end;i++){
			errorbuf[i]  = error[i][korder-2];
			resultbuf[i] = result[i][korder-2];
		}

#ifdef PARALLEL	
		MPI_Allreduce(MPI_IN_PLACE,errorbuf, NumDynVariables,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE,resultbuf,NumDynVariables,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
#endif

		for(int i=0;i<NumDynVariables;i++){
                        error[i][korder-2]  = errorbuf[i];
                        result[i][korder-2] = resultbuf[i];
                }

		double gbs_error = DBL_MAX;
		int index=-1;

		double gbs_tolerance = ComputationToDoList.ComputationalTask[SubStepDivision[0]].ThisR->gbs_tolerance;

		for(int i=0;i<korder-1;i++){
			maxerror[i] = 1e10;
		}

		for(int i=korder-2;i<korder-1;i++){
			double error_in_this_level = -1;

			for(int j=0;j<NumDynVariables;j++){

				if( error[j][i] >= error_in_this_level ){
					error_in_this_level = error[j][i];
				}
			}

			if( error_in_this_level <= gbs_error ){
				gbs_error = error_in_this_level;
				index = i;
			}

			maxerror[i] =  error_in_this_level;

		}

		int flag = 0;
		if( gbs_error <= gbs_tolerance ){
			flag = 1;
		}

		double km = index+2;

                double Hnextfac = 0.94*pow( 0.65/(gbs_error/gbs_tolerance), 1.0 / (2*km+1));

		double H = ThisR->Hstep;

		if( gbs_error <= gbs_tolerance ){

			//if(ThisTask==0) printf("error %e/%e --> converged --- %d\n",gbs_error, gbs_tolerance, NumDynVariables);

			status = 1;

			for(int v=0;v<NumDynVariables;v++){
				ThisR->State[v] = result[v][index];
			}
			if( Hnextfac > 5.0 ){
                        	Hnextfac = 5.0;
                	}

			double Hprev = ThisR->Hstep;
			ThisR->Hstep *= Hnextfac;
			into_Cartesian_from_MST(ThisR);

			ok_steps++;


#if 0

			analysis_for_output( ThisR, &a1, &e1, &theta1, &EN1, &Lz1 );


			double dEN     = fabs( (EN1-EN0)/EN0 );
			double dLz    = fabs((Lz1-Lz0)/Lz0);
			double dtheta = fabs(atan2(sin(theta1-theta0), cos(theta1-theta0)));
			double da     = fabs((a1-a0)/a0);
			double de     = fabs((e1-e0)/e0);

			wtime1 = clock();
			double elapsed = (double) (wtime1-wtime0)/CLOCKS_PER_SEC;


			if(ThisTask==0){
	                       	double time = ThisR->State[ThisR->TimeIndex];
                        	//fprintf(output,"%.20e %.20e %.20e %.20e %.20e %.20e %.20e %.20e %.20e %.10e\n",time,gbs_error,dEN,dLz,dtheta,da,de,Hprev,elapsed,1.0*failed_steps/(ok_steps+failed_steps));
				//printf("t %.10e gbs %.10e err %.10e Lz %.10e theta %.10e a %.10e e %.10e H %.10e walltime %.10e\n",time,gbs_error,dEN,dLz,dtheta,da,de,Hprev,elapsed);
				//fflush(output);
                	}

#endif

#if 0
			if(ThisTask==0){

				double x1 = ThisR->Pos[3*0+0];
                                double y1 = ThisR->Pos[3*0+1];
                                double x2 = ThisR->Pos[3*1+0];
                                double y2 = ThisR->Pos[3*1+1];
                                double x3 = ThisR->Pos[3*2+0];
                                double y3 = ThisR->Pos[3*2+1];

				double vx1 = ThisR->Vel[3*0+0];
                                double vy1 = ThisR->Vel[3*0+1];
                                double vx2 = ThisR->Vel[3*1+0];
                                double vy2 = ThisR->Vel[3*1+1];
                                double vx3 = ThisR->Vel[3*2+0];
                                double vy3 = ThisR->Vel[3*2+1];
				double m1 = 0.03,m2=0.04,m3=0.05;

				double rvec[3] = {0}, vvec[3] = {0}, r2=0,v2=0,rdotv=0;
				rvec[0] = x3-x2;	rvec[1] = y3-y2;	rvec[2] = 0;
				vvec[0] = vx3-vx2;        vvec[1] = vy3-vy2;        vvec[2] = 0;
				for(int k=0;k<3;k++){
					r2 += rvec[k]*rvec[k];
					v2 += vvec[k]*vvec[k];
					rdotv += rvec[k]*vvec[k];
				}
				double r = sqrt(r2);

				double beta = atan2(y1,x1)*180/M_PI;

				double GM = GCONST*(m2+m3);
                		double a = 1.0/(2.0/r - v2/GM);
                		double ecc[3];
                		double e=0;
                		for(int k=0;k<3;k++){
                        		ecc[k] = (1.0/r-1.0/(a))*rvec[k] - rdotv*vvec[k]/GM;
                        		e += ecc[k]*ecc[k];
                		}
                		e = sqrt(e);
                                double time = ThisR->State[ThisR->TimeIndex];

                                fprintf(output,"%.20e %.20e %.20e %.20e %.20e %.20e %.20e %.20e %.20e %.20e\n",time,x1,y1,x2,y2,x3,y3,beta,a,e);
                                fflush(output);
                        }
#endif


		} else {

			//if(ThisTask==0) printf("error %e/%e --> diverged, dH %e\n",gbs_error, gbs_tolerance, ThisR->Hstep);

			status = 0;
			double Hprev = ThisR->Hstep;

                        //into_Cartesian_from_MST(ThisR);
                        //analysis_for_output( ThisR, &a1, &e1, &theta1, &EN1, &Lz1 );

                        //double dEN     = fabs( (EN1-EN0)/EN0 );
                        //double dLz    = fabs((Lz1-Lz0)/Lz0);
                        //double dtheta = fabs(atan2(sin(theta1-theta0), cos(theta1-theta0)));
                        //double da     = fabs((a1-a0)/a0);
                        //double de     = fabs((e1-e0)/e0);

                        //wtime1 = clock();
                        //double elapsed = (double) (wtime1-wtime0)/CLOCKS_PER_SEC;

                        //if(ThisTask==0){
                            //    double time = ThisR->State[ThisR->TimeIndex];
                                //fprintf(output,"%.20e %.20e %.20e %.20e %.20e %.20e %.20e %.20e %.20e\n",time,gbs_error,dEN,dLz,dtheta,da,de,Hprev,elapsed);
                                //printf("t %.10e gbs %.10e err %.10e Lz %.10e theta %.10e a %.10e e %.10e H %.10e walltime %.10e\n",time,gbs_error,dEN,dLz,dtheta,da,de,Hprev,elapsed);
                                //fflush(output);
                        //}

			failed_steps++;

			if( Hnextfac > 0.7 ){
                        	Hnextfac = 0.7;
                	}

			ThisR->Hstep *= Hnextfac;
		}
#if 0
		for(int i=0;i<NumDynVariables;i++){
                        free(error[i]);			total_memory -= (korder-1)*sizeof(double);
                        free(result[i]);		total_memory -= (korder-1)*sizeof(double);
                }
                free(error);				total_memory -= NumDynVariables*sizeof(double);
                free(result);				total_memory -= NumDynVariables*sizeof(double);
#endif


		free(SubStepDivision);			total_memory -= korder*sizeof(int);


#if 0
		free(y);				total_memory -= korder*sizeof(double);
		free(yscal);				total_memory -= korder*sizeof(double);
                free(errorbuf);
                free(resultbuf);
#endif

	}

        clock_t mytime = clock();
        text3 += (double) (mytime-aika0)/CLOCKS_PER_SEC;

	measure_cpu_time( CPU_TIME_GBS );

	return status;

}

// Integrate the subsystems using AR-MST for the desired time interval
void run_integrator( struct RegularizedRegion *R, double time_interval, int NumberOfSystems ){

#if 0


                        if(ThisTask==0){

                                double x1 = R->Pos[3*0+0];
                                double y1 = R->Pos[3*0+1];
                                double x2 = R->Pos[3*1+0];
                                double y2 = R->Pos[3*1+1];
                                double x3 = R->Pos[3*2+0];
                                double y3 = R->Pos[3*2+1];

                                double vx1 = R->Vel[3*0+0];
                                double vy1 = R->Vel[3*0+1];
                                double vx2 = R->Vel[3*1+0];
                                double vy2 = R->Vel[3*1+1];
                                double vx3 = R->Vel[3*2+0];
                                double vy3 = R->Vel[3*2+1];
                                double m1 = 0.03,m2=0.04,m3=0.05;

                                double rvec[3] = {0}, vvec[3] = {0}, r2=0,v2=0,rdotv=0;
                                rvec[0] = x3-x2;        rvec[1] = y3-y2;        rvec[2] = 0;
                                vvec[0] = vx3-vx2;        vvec[1] = vy3-vy2;        vvec[2] = 0;
                                for(int k=0;k<3;k++){
                                        r2 += rvec[k]*rvec[k];
                                        v2 += vvec[k]*vvec[k];
                                        rdotv += rvec[k]*vvec[k];
                                }
                                double r = sqrt(r2);

                                double beta = atan2(y1,x1)*180/M_PI;

                                double GM = GCONST*(m2+m3);
                                double a = 1.0/(2.0/r - v2/GM);
                                double ecc[3];
                                double e=0;
                                for(int k=0;k<3;k++){
                                        ecc[k] = (1.0/r-1.0/(a))*rvec[k] - rdotv*vvec[k]/GM;
                                        e += ecc[k]*ecc[k];
                                }
                                e = sqrt(e);
                                double time = R->State[R->TimeIndex];

                                fprintf(output,"%.20e %.20e %.20e %.20e %.20e %.20e %.20e %.20e %.20e %.20e\n",time,x1,y1,x2,y2,x3,y3,beta,a,e);
                                fflush(output);
                        }

#endif


	// Needed variables
	double dt = 1e-7;
	double Hstep;
	double Hnext;

	// Initialize systems to integrate
	for(int i=0;i<NumberOfSystems;i++){
		compute_total_energy( &R[i] );
		R[i].Hstep = dt*R[i].U;
		R[i].State[R[i].TimeIndex] = 0;
	}

	double percent = time_interval/50;
	int epoch = -1;

	// Now proceed to integration
	int not_finished = 1;
	int counter = 0;
	int redo = 1;
	int status;
	int repeat = 0;

	do {

		do {

			redo = 0;

			// Divide the systems and substep divisions for tasks
			divide_computational_load( R, NumberOfSystems );

			// Perform the leapfrog tasks
			for(int ctask=0;ctask<ComputationToDoList.NumberOfComputationalTasks;ctask++){
				if( ThisGbsGroup == ComputationToDoList.ComputationalTask[ctask].GroupToPerformTask ){

					// copy the system in order not to overwrite the original one
					struct RegularizedRegion *ThisR = ComputationToDoList.ComputationalTask[ctask].ThisR;

					copy_regularized_region( ThisR, &ComputationToDoList.CopyOfSingleRegion[ctask] );

					// leapfrog the substep division
					Hstep = ComputationToDoList.CopyOfSingleRegion[ctask].Hstep;

					int flag = 0;
					clock_t timing0 = clock();
					mst_leapfrog( &ComputationToDoList.CopyOfSingleRegion[ctask], Hstep, ComputationToDoList.ComputationalTask[ctask].NumberOfSubsteps, &flag );
					clock_t timing1 = clock();

#ifdef SERIALALALA
					if( flag == 1 ){
						ThisR->Hstep *= 0.5;
						//failed_steps++;
						redo = 1;
						break;
					}
#endif

				}
			}

		} while(redo);

my_barrier();
clock_t aika0= clock(),aika1;

#ifdef PARALLEL
		// We need to communicate the results
		for(int ctask=0;ctask<ComputationToDoList.NumberOfComputationalTasks;ctask++){
			int root 	= ComputationToDoList.ComputationalTask[ctask].GroupToPerformTask*NumTaskPerGbsGroup;
			double *commbuf = ComputationToDoList.CopyOfSingleRegion[ctask].State;

			int count       = ComputationToDoList.CopyOfSingleRegion[ctask].NumDynVariables;
			if(ThisTask != root){
				for(int i=0;i<count;i++){
					commbuf[i] = 0;
				}
			}
			MPI_Bcast( commbuf, count, MPI_DOUBLE, root, MPI_COMM_WORLD );
			//MPI_Allreduce(  MPI_IN_PLACE, commbuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
		}
		aika1 = clock();

		time_gbscomm += (double) (aika1-aika0)/CLOCKS_PER_SEC;
#endif

		// Now preform the GBS extrapolation, check convergence and obtain next timestep

		my_barrier();
		aika0 = clock();
		status = extrapolate_all_variables( NumberOfSystems, KMAX );
		aika1 = clock();
		time_gbs += (double) (aika1-aika0)/CLOCKS_PER_SEC;

		if( status == 1 ){
		repeat = 0;

		// Check whether we are ready
		double time = R[0].State[R[0].TimeIndex];
		if( time > time_interval ){
			not_finished = 0;
		} else {

			int old_epoch = epoch;
			epoch = (int) floor( time/percent );
			if( epoch != old_epoch ){
#ifdef OLDPRIM
				PrimMST(&R[0]);
#else
				DivideAndConquerPrim( &R[0] );				
#endif
			}
		}

#if 0
		if(ThisTask==0) printf("time %e --- dt %.10e\n",time,R[0].Hstep);
#endif

		} else {
			repeat = 1;
		}


	} while( not_finished  );


}

void allocate_regularized_region(struct RegularizedRegion *S, int N){

        S->gbs_tolerance         = GBSTOL;

        S->output_time_tolerance = 1e-6;
	S->NumVertex	         = N;

        S->NumEdge         = (N*(N-1))/2;
        S->NumDynVariables = 6*(N-1)+2;

        S->TimeIndex       = S->NumDynVariables-1;
        S->BIndex          = S->NumDynVariables-2;

        S->Pos          = calloc(3*N,     sizeof(double));
        S->Vel          = calloc(3*N,     sizeof(double));
        S->State        = calloc(S->NumDynVariables, sizeof(double));
        S->Mass         = calloc(N,       sizeof(double));
        S->Acc          = calloc(3*N,     sizeof(double));
        S->MSTedgeAcc   = calloc(3*(N-1), sizeof(double));		total_memory += (3*N+3*N+2*S->NumDynVariables+N+3*N+3*(N-1))*sizeof(double);

        S->AccPN          = calloc(3*N,     sizeof(double));
        S->AuxVel         = calloc(3*N,     sizeof(double));
        S->AuxEdgeVel     = calloc(3*(N-1), sizeof(double));
        S->MSTedgeAcc_PN  = calloc(3*(N-1), sizeof(double));		total_memory += ( 2*3*N + 2*3*(N-1) )*sizeof(double);

        S->Vertex          = calloc(N,          sizeof(struct GraphVertex));	total_memory += N*sizeof(struct GraphVertex);
        
	double safetybuf = 1.05;
	int NumLocalEdge = (int) floor( safetybuf*S->NumEdge/Ntask );
	S->LocalEdge       = calloc(NumLocalEdge, sizeof(struct GraphEdge));	total_memory += NumLocalEdge*sizeof(struct GraphEdge);
	S->LocalEdgeSubset = calloc(NumLocalEdge, sizeof(struct GraphEdge));
	S->EdgeInMST       = calloc(N-1, sizeof(struct GraphEdge));

}


void allocate_armst_structs( struct RegularizedRegion **R, int NumberOfSystems, int MaxNumPart ){

	int MaxDynamicalVariables = 6*(MaxNumPart-1)+2;
	*R = calloc(NumberOfSystems,sizeof(struct RegularizedRegion));	total_memory += NumberOfSystems*sizeof(struct RegularizedRegion);

        for(int i=0;i<NumberOfSystems;i++){
                allocate_regularized_region( R[i], MaxNumPart );
        }

        gbsS = calloc(KMAX,sizeof(double *));                 total_memory += KMAX*sizeof(double *);
        for(int i=0;i<KMAX;i++){
                gbsS[i] = calloc(KMAX,sizeof(double));        total_memory += KMAX*sizeof(double);
        }
	int Nbuf = 5;
        fastindex  = calloc(Nbuf*Ntask,sizeof(int));
        fastv1     = calloc(Nbuf*Ntask,sizeof(int));
        fastv2     = calloc(Nbuf*Ntask,sizeof(int));
        fastweight = calloc(Nbuf*Ntask,sizeof(double));

	int korder = 8;
	int NumDynVariables = MaxDynamicalVariables;

	        errorbuf  = calloc(NumDynVariables,sizeof(double));
                resultbuf = calloc(NumDynVariables,sizeof(double));

                y = calloc(korder,sizeof(double));              total_memory += korder*sizeof(double);
                yscal = calloc(korder,sizeof(double));          total_memory += korder*sizeof(double);
                maxerror = calloc(korder-1,sizeof(double));     total_memory += (korder-1)*sizeof(double);

                error = calloc(NumDynVariables,sizeof(double *));       total_memory += NumDynVariables*sizeof(double *);
                result = calloc(NumDynVariables,sizeof(double *));      total_memory += NumDynVariables*sizeof(double *);
                for(int i=0;i<NumDynVariables;i++){
                        error[i] = calloc(korder-1,sizeof(double));     total_memory += (korder-1)*sizeof(double);
                        result[i] = calloc(korder-1,sizeof(double));    total_memory += (korder-1)*sizeof(double);
                }

	indexlist = calloc(8,sizeof(int *));
	for(int i=0;i<8;i++){
		indexlist[i] = calloc(MaxNumPart, sizeof(int));
	}
	ParticleInNode = calloc(MaxNumPart,sizeof(struct OctreeNode *));

	//double mtot = 0;
        //MPI_Allreduce(&total_memory,&mtot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        //if(ThisTask==0) printf("Currently allocated memory: %.10lf GB\n",mtot/(1024*1024*1024));

	int NumAlloc = NumberOfSystems*KMAX;

        ComputationToDoList.ComputationalTask = calloc(NumAlloc,sizeof(struct ComTask));	total_memory += NumAlloc*KMAX*sizeof(struct ComTask);
        for(int i=0;i<NumAlloc;i++){
                ComputationToDoList.ComputationalTask[i].ResultState = calloc(MaxDynamicalVariables, sizeof(double));	total_memory += MaxDynamicalVariables*sizeof(double);
        }
        ComputationToDoList.CopyOfSingleRegion = calloc(NumAlloc,sizeof(struct RegularizedRegion));	total_memory += NumAlloc*KMAX*sizeof(struct RegularizedRegion);

        for(int i=0;i<NumAlloc;i++){
                allocate_regularized_region( &ComputationToDoList.CopyOfSingleRegion[i], MaxNumPart );	
        }

	//mtot = 0;
        //MPI_Allreduce(&total_memory,&mtot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        //if(ThisTask==0) printf("Currently allocated memory: %.10lf GB\n",mtot/(1024*1024*1024));

}

void read_ic_from_file( struct RegularizedRegion *R, char *INPUTFILE ){

        // Read in the test IC
        FILE *fp;
        for(int task=0;task<Ntask;task++){
		my_barrier();

        if(ThisTask==task){
                fp = fopen(INPUTFILE,"r");
                //for(int i=0;i<R->NumVertex;i++){
		for(int i=R->NumVertex-1;i>-1;i--){
int s = fscanf(fp,"%d %lf %lf %lf %lf %lf %lf %lf\n", &R->Vertex[i].type, &R->Mass[i], &R->Pos[3*i+0], &R->Pos[3*i+1], &R->Pos[3*i+2], &R->Vel[3*i+0], &R->Vel[3*i+1], &R->Vel[3*i+2]);
		//R->Pos[3*i+2] = 0;
                }
                fclose(fp);
        }
		my_barrier();
        }

}

void generate_binary_ic( struct RegularizedRegion *R, double Mbinary, double semimajor, double eccentric, double theta, double *Pkepler){

	//double angle = (double) (1.0* (rand()%360));
	//theta = angle;

        double  x1 = +semimajor*(1+eccentric)/2;
        double  x2 = -semimajor*(1+eccentric)/2;
        double  z1 = 0;
        double  z2 = 0;

        double y1 = x1*sin(theta);
        x1 *= cos(theta);
        double y2 = x2*sin(theta);
        x2 *= cos(theta);
        double vy1 = +0.5*sqrt( (1-eccentric)/(1+eccentric) * GCONST*Mbinary/semimajor );
        double vy2 = -0.5*sqrt( (1-eccentric)/(1+eccentric) * GCONST*Mbinary/semimajor );
        double vx1 = -vy1*sin(theta);
        vy1 *= cos(theta);
        double vx2 = -vy2*sin(theta);
        vy2 *= cos(theta);
        double vz1 = 0;
        double vz2 = 0;

        R->Vertex[0].type = 5;
        R->Vertex[0].id   = 0;
        R->Mass[0]        = Mbinary/2;
        R->Pos[3*0+0]     = x1;
        R->Pos[3*0+1]     = y1;
        R->Pos[3*0+2]     = z1;
        R->Vel[3*0+0]     = vx1;
        R->Vel[3*0+1]     = vy1;
        R->Vel[3*0+2]     = vz1;
        R->Vertex[1].type = 5;
        R->Mass[1]        = Mbinary/2;
        R->Vertex[1].id   = 1;
        R->Pos[3*1+0]     = x2;
        R->Pos[3*1+1]     = y2;
        R->Pos[3*1+2]     = z2;
        R->Vel[3*1+0]     = vx2;
        R->Vel[3*1+1]     = vy2;
        R->Vel[3*1+2]     = vz2;

        *Pkepler = 2*M_PI*sqrt( semimajor*semimajor*semimajor/(GCONST*Mbinary));


}


void generate_pythagorean_triplet( struct RegularizedRegion *R ){

	// point 1
	R->Vertex[0].type = 5;
        R->Vertex[0].id   = 0;
        R->Mass[0]        = 0.03;	// m
        R->Pos[3*0+0]     = 0;		// x
        R->Pos[3*0+1]     = 0.04;	// y
        R->Pos[3*0+2]     = 0;
        R->Vel[3*0+0]     = 0;
        R->Vel[3*0+1]     = 0;
        R->Vel[3*0+2]     = 0;
        // point 2
        R->Vertex[1].type = 5;
        R->Vertex[1].id   = 1;
        R->Mass[1]        = 0.04;	// m
        R->Pos[3*1+0]     = -0.03;	// x
        R->Pos[3*1+1]     = 0;		// y
        R->Pos[3*1+2]     = 0;
        R->Vel[3*1+0]     = 0;
        R->Vel[3*1+1]     = 0;
        R->Vel[3*1+2]     = 0;
        // point 3
        R->Vertex[2].type = 5;
        R->Vertex[2].id   = 2;
        R->Mass[2]        = 0.05;	// m
        R->Pos[3*2+0]     = 0;		// x
        R->Pos[3*2+1]     = 0;		// y
        R->Pos[3*2+2]     = 0;
        R->Vel[3*2+0]     = 0;
        R->Vel[3*2+1]     = 0;
        R->Vel[3*2+2]     = 0;


}



int main(int argc, char *argv[]){

	initialize_mpi_or_serial();

	my_barrier();

        GBSTOL    = atof(argv[3]);
        MAXPART   = atoi(argv[2]);

	total_memory = 0;

for(int loop=0;loop<=0;loop++){

	char fname[99];
	char str[99];
	sprintf(str, "%.4d", loop);
	sprintf(fname,"%s%s%s", "output/output_",str,".txt");

	if(ThisTask==0){
		//printf("%s %s\n",argv[1],argv[2]);
		//output = fopen(fname,"w");
		//fflush(stdout);
	}

	Niter = 0;
	Nredo = 0;
	MyC = 0;
	realtime = 0;
	fictitioustime=0;
	failed_steps = 0;
	ok_steps = 0;
	time_force = 0;
	time_comm  = 0;
	sum_order = 0;
	time_mst = 0;
	time_gbs = 0;
	time_gbscomm = 0;
	time_edge = 0;
	time_coord1 = 0;
	time_coord2 = 0;
	time_coord3 = 0;
	time_coord4 = 0;
	tprim1=0,tprim2=0,tprim3=0,tprim4=0,tprim5=0,tprim6=0,tprim7=0;
	text1=0,text2=0,text3=0;
	primfail=0,primsucc=0;
	loopt1=0,loopt2=0,loopt3=0,loopt4=0;

	my_barrier();

	/////////////////////////////////////////
	struct RegularizedRegion *R;
        int NumberOfSystems = 1;
        int MaxNumPart = MAXPART;

	allerrors = calloc(6*(MaxNumPart-1)*2,sizeof(double));
	allocate_armst_structs(&R,NumberOfSystems,MaxNumPart);

	read_ic_from_file( &R[0], argv[1] );

	my_barrier();

	previous_time = clock();
        current_time = clock();
	cpu_times = calloc(5,sizeof(clock_t));

	double IntegrationTime = 1e-4;

	into_CoM_frame( &R[0] );

	// Build the MST and integrate for some time interval

	wtime0 = clock();


#ifdef OLDPRIM
	PrimMST(&R[0]);
#else
	DivideAndConquerPrim( &R[0] );
#endif


	//analysis_for_output( &R[0], &a0, &e0, &theta0, &EN0, &Lz0 );
	//analysis_for_smallest_period_around_bh( &R[0] );

	compute_total_energy( &R[0] );
	double mye0 = R[0].H;

	double a0,a1;
	get_scaling( R[0].NumVertex, R[0].Mass, &a0,&a1, R[0].T, R[0].U );

	//analysis_for_smallest_binary_period( &R[0] );


	run_integrator(R,IntegrationTime,1);

	compute_total_energy( &R[0] );

        double mye1 = R[0].H;
	double myerr = fabs((mye0-mye1)/mye0);

	//analysis_for_smallest_period_around_bh( &R[0] );

	wtime1 = clock();
	double elapsed = (double) (wtime1-wtime0)/CLOCKS_PER_SEC;

	measure_cpu_time( CPU_TIME_MISC );

	double cputimes[5];
	double tot_time = 0;
	for(int i=0;i<5;i++){
		cputimes[i] = (double) (1.0*cpu_times[i]) / CLOCKS_PER_SEC;
		tot_time += cputimes[i];
	}	
	double tfrac[4];
	for(int i=1;i<5;i++){
		tfrac[i-1] = cputimes[i]/tot_time;
	}

	double time_coord = time_coord1 +time_coord2+time_coord3+time_coord4;

	if(ThisTask==0){
		printf("%d %d %d %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n",Ntask,MAXPART,ok_steps+failed_steps,time_force-time_comm,time_comm,elapsed,time_mst,time_gbs,time_gbscomm,time_coord);

#if 0
		printf("force:     %lf\n",100*(time_force-time_comm)/elapsed);
		printf("forcecomm: %lf\n",100*(time_comm)/elapsed);
		printf("prim:      %lf\n",100*(time_mst)/elapsed);
		printf("gbs:       %lf\n",100*(time_gbs)/elapsed);
		printf("divcomm:   %lf\n",100*(time_gbscomm)/elapsed);
		printf("coord  :   %lf\n",100*(time_coord)/elapsed);

		printf("\n--- prim ---\n");
#ifdef OLDPRIM
		printf("%lf %lf %lf %lf %lf\n",100*tprim1/elapsed,100*tprim2/elapsed,100*(tprim3)/elapsed,100*tprim4/elapsed,100*tprim5/elapsed);
#else
		printf("octree           %lf\n",100*tprim1/elapsed);
		printf("octree particles %lf\n",100*tprim2/elapsed);
		printf("sub-msts         %lf\n",100*tprim3/elapsed);
		printf("meta-mst         %lf ---> %lf and %lf\n",100*tprim4/elapsed, 100*tprim6/elapsed, 100*tprim7/elapsed);
		printf("combine msts     %lf\n",100*tprim5/elapsed);
		printf("meta loops: %lf %lf %lf %lf\n", loopt1,loopt2,loopt3,loopt4);
#endif
		printf("\n--- exrt ---\n");
                printf("%lf %lf %lf\n",100*text1/elapsed,100*text2/elapsed,100*text3/elapsed);

		fflush(stdout);
#endif
	}

	if(ThisTask==0){
                //fclose(output);
	}


	// Free stuff

        for(int i=0;i<NumberOfSystems*KMAX;i++){
                free(ComputationToDoList.ComputationalTask[i].ResultState);
        }
	free(ComputationToDoList.ComputationalTask);

	free(R[0].LocalEdge);
	free(R[0].Vertex);
	free(R[0].Pos);
	free(R[0].Vel);
	free(R[0].Acc);
	free(R[0].Mass);
	free(R[0].State);
	free(R[0].MSTedgeAcc);
	free(R[0].EdgeInMST);
	free(gbsS);

}	// end loop

#ifdef PARALLEL
	MPI_Finalize();
#endif

	return 0;
}
