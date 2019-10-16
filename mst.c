#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <limits.h>
#include "regularization.h"
#include <string.h>
#include <unistd.h>

#define mydouble double

int extrapolate_all_variables( int NumberOfSystems, int korder );

double GBSTOL;
double ETA;

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

double sum_order;
double time_force;

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

void my_barrier(void){

#ifdef PARALLEL
        MPI_Barrier(MPI_COMM_WORLD);
#endif

}

void die(void){

	my_barrier();
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

#ifdef PARALLEL
void initialize_node_comm(void){

	MPI_Group WORLD_GROUP, GBS_GROUP;
	MPI_Comm_group(MPI_COMM_WORLD, &WORLD_GROUP);
	int *myranks;

	ThisGbsGroup         = (int) floor(ThisTask/NumTaskPerGbsGroup);
	ThisTask_in_GbsGroup = ThisTask%NumTaskPerGbsGroup;

        myranks = calloc(NumTaskPerGbsGroup, sizeof(int));	total_memory += NumTaskPerGbsGroup*sizeof(int);

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
    	free(myranks);		total_memory -= NumTaskPerGbsGroup*sizeof(int);

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

	NumGbsGroup        = 8;
        NumTaskPerGbsGroup = (int) Ntask/NumGbsGroup;

	if( NumGbsGroup*NumTaskPerGbsGroup != Ntask ){
		if(ThisTask==0){
			printf("NumGbsGroup*NumTaskPerGbsGroup != Ntask\n");
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

get_v1_v2( int N, int target, int *v1, int *v2){

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

void PrimMST(struct RegularizedRegion *R){

	measure_cpu_time( CPU_TIME_MISC );

	int ThisBlock,CumNumEdge=0,istart=0,jstart;
#ifdef PARALLEL
	loop_scheduling_N2(R->NumVertex,Ntask,&istart,&jstart,&ThisBlock,&CumNumEdge,ThisTask);
#endif
	int NumLocalEdges=0; int loop=1; int c=0;

	double rmin=1e10,r2;
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
			if(c == ThisBlock ){
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
	int MinIndex;

	for(int i=0;i<R->NumVertex;i++){
		R->Vertex[i].id = i;
		R->Vertex[i].degree = 0;
		R->Vertex[i].edgetoparent = -1;
		R->Vertex[i].parent = -1;
		R->Vertex[i].level  = 0;
		R->Vertex[i].inMST  = 0;
	}

        R->Vertex[last].inMST = 1;

#ifdef CHAIN

	int head = -1;
	int tail = -1;
	double my_weight = DBL_MAX;
	wi.weight 	 = DBL_MAX;
        wi.index  	 = -1;
	R->Vertex[last].inMST = 0;
	for(int i=0;i<NumLocalEdges;i++){
              	if( my_weight > R->LocalEdge[i].weight){
			my_weight = R->LocalEdge[i].weight;
        	       	wi.weight = R->LocalEdge[i].weight;
                        wi.index  = R->Vertex[R->LocalEdge[i].vertex1].id;
	        }
        }
#ifdef PARALLEL
       	MPI_Allreduce( MPI_IN_PLACE, &wi, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD );
#endif
	for(int i=0;i<R->NumVertex;i++){
		if( R->Vertex[i].id == wi.index ){
			R->Vertex[i].inMST = 1;
			head = i;
			break;
		}
	}
#endif
	c=0;
	do {

		wi.weight = 1;
		wi.index  = -1;
		for(int i=0;i<NumLocalEdges;i++){
#ifdef CHAIN
			int headtailcount = 0;
			if( R->LocalEdge[i].vertex1 == head || R->LocalEdge[i].vertex1 == tail ){
				headtailcount++;
			}
			if( R->LocalEdge[i].vertex2 == head || R->LocalEdge[i].vertex2 == tail ){
                                headtailcount++;
                        }
			if( headtailcount < 1 ){
				continue;
			}
			if( headtailcount > 1 ){
                                continue;
                        }
#endif
			if( R->Vertex[R->LocalEdge[i].vertex1].inMST != R->Vertex[R->LocalEdge[i].vertex2].inMST ){
				wi.weight = R->LocalEdge[i].weight;
                               	wi.index  = R->LocalEdge[i].id;
				break;
			}
		}
#ifdef PARALLEL
		MPI_Allreduce( MPI_IN_PLACE, &wi, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD );
#endif

		R->MSTedgeList[c] = wi.index;

		int v1,v2;	
		get_v1_v2( R->NumVertex, wi.index, &v1,&v2 );

		R->Vertex[v1].degree++;
		R->Vertex[v2].degree++;

#ifdef CHAIN
		if(tail == -1){
			if( v1 == head ){
				tail = v2;
			} else {
				tail = v1;
			}
		}
#endif
		if( R->Vertex[v1].inMST == 0  ){
#ifdef CHAIN
			if(v2==head){
				head = v1;
			} else {
				tail = v1;
			}
#endif

			R->Vertex[v1].parent = v2;
			R->Vertex[v1].level  = R->Vertex[v2].level+1;
			R->Vertex[v1].inMST  = 1;
			R->Vertex[v1].edgetoparent = c;
			for(int k=0;k<3;k++){
				R->State[3*c+k] 			  = R->Pos[3*v1+k] - R->Pos[3*v2+k];
                                R->State[3*(R->NumVertex-1)+3*c+k] 	  = R->Vel[3*v1+k] - R->Vel[3*v2+k];
			}
		} else {
#ifdef CHAIN
			if(v1==head){
                                head = v2;
                        } else {
                                tail = v2;
                        }
#endif
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

	measure_cpu_time( CPU_TIME_PRIM );

}

void compute_U(struct RegularizedRegion *R){

	measure_cpu_time( CPU_TIME_MISC );

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

#ifdef PARALLEL
	if( NumTaskPerGbsGroup > 1 ){
		MPI_Allreduce(MPI_IN_PLACE,&R->U,   1,MPI_DOUBLE,MPI_SUM,MPI_COMM_GBS_GROUP);
	}
#endif

	if(!isfinite(R->U)){
		printf("U not finite.\n");
		exit(0);
	}

	R->U *= GCONST;

	measure_cpu_time( CPU_TIME_POTENTIALENERGY );

}

double estimate_timestep( struct RegularizedRegion *R ){

	double min_dt = DBL_MAX;
	for(int j=0;j<R->NumVertex;j++){
		double dt = ETA*sqrt( R->State[j]/R->MSTedgeAcc[j] );
		if( dt < min_dt  ){
			min_dt = dt;
		}
	}
	return min_dt;
}

void compute_U_and_Newtonian_Acc(struct RegularizedRegion *R){

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

#ifdef PARALLEL
        if( NumTaskPerGbsGroup > 1 ){
                MPI_Allreduce(MPI_IN_PLACE,R->Acc,  3*R->NumVertex,MPI_DOUBLE,MPI_SUM,MPI_COMM_GBS_GROUP);
                MPI_Allreduce(MPI_IN_PLACE,&R->U,   1,             MPI_DOUBLE,MPI_SUM,MPI_COMM_GBS_GROUP);
        }
#endif

	int hi,lo;
        for(int i=0;i<R->NumVertex-1;i++){
                int index = R->MSTedgeList[i];

                int v1,v2;
                get_v1_v2( R->NumVertex, index, &v1,&v2 );

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

	measure_cpu_time( CPU_TIME_ACC );
}


void into_Cartesian_from_MST(struct RegularizedRegion *R){

	int index = R->MSTedgeList[0];

	int v1,v2;
        get_v1_v2( R->NumVertex, index, &v1,&v2 );

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
		index = R->MSTedgeList[i];

                get_v1_v2( R->NumVertex, index, &v1,&v2 );

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
}

void into_CoM_frame(struct RegularizedRegion *R){

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

        int index = R->MSTedgeList[0];

	int v1,v2;
        get_v1_v2( R->NumVertex, index, &v1,&v2 );

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
                index = R->MSTedgeList[i];

                get_v1_v2( R->NumVertex, index, &v1,&v2 );

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
}

void update_vel(struct RegularizedRegion *R, double *Vel, enum velocity_type propagated_velocity){

        int index = R->MSTedgeList[0];

	int v1,v2;
        get_v1_v2( R->NumVertex, index, &v1,&v2 );

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
                index = R->MSTedgeList[i];

                get_v1_v2( R->NumVertex, index, &v1,&v2 );

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
		for(int i=0;i<KMAX;i++){
			int ThisSubdivisionCost = get_n_int(i) *N*(N-1)/2;
			int GroupIndex = gbs_group_with_min_workload(GbsGroupWorkLoad);
			GbsGroupWorkLoad[GroupIndex] += ThisSubdivisionCost;
			ComputationToDoList.ComputationalTask[counter].WhichRegion = region;
			ComputationToDoList.ComputationalTask[counter].GroupToPerformTask = GroupIndex;
			ComputationToDoList.ComputationalTask[counter].ThisR = &R[region];
			ComputationToDoList.ComputationalTask[counter].NumberOfSubsteps = get_n_int(i);
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

#if 1

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
                Copy->MSTedgeList[j] = Orig->MSTedgeList[j];
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

	double **S;
	S = calloc(N,sizeof(double *));			total_memory += N*sizeof(double *);
        for(int i=0;i<N;i++){
        	S[i] = calloc(N,sizeof(double));	total_memory += N*sizeof(double);
        }

        for(int i=0;i<N;i++){
                S[i][0] = y[i];
        }
        for(int j=1;j<N;j++){
                for(int i=0;i<N-j;i++){
                        double nterm = pow(get_n_double(i+j)/get_n_double(i),2);
                        S[i][j] = ( nterm*S[i+1][j-1] - S[i][j-1] ) / (nterm-1.0);
                }
        }
        for(int i=1;i<N;i++){
                result[i-1] = S[0][i];
                error[i-1]  = fabs(S[0][i]-S[0][i-1]) / yscal[i];
        }

	for(int i=0;i<N;i++){
		free(S[i]);	total_memory -= N*sizeof(double);
	}
	free(S);		total_memory -= N*sizeof(double *);

}

void get_scaling( int N, double *Mass, double *R, double *V, double T, double U ){

	double M = 0;

	for(int i=0;i<N;i++){
		M += Mass[i];
	}
	*V = sqrt(2*T/M);
	*R = fabs(GCONST*M*M/(2*U));

}

int extrapolate_all_variables( int NumberOfSystems, int korder ){

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

		double rscale,vscale;
		get_scaling( ThisR->NumVertex,ThisR->Mass,&rscale,&vscale,ThisR->T,ThisR->U);

		for(int v=0;v<NumDynVariables;v++){
			for(int i=0;i<korder;i++){
				y[i]     =  ComputationToDoList.CopyOfSingleRegion[ SubStepDivision[i] ].State[v];
				yscal[i] = fabs(y[i]);

#if 0
				if( v<3*(ThisR->NumDynVariables-1) ){
					yscal[i] = rscale;
				} else if( v<6*(ThisR->NumDynVariables-1) ){
					yscal[i] = vscale;
				} else {
	                                yscal[i] = fabs(y[i]);
				}
#endif

			}
			extrapolate_single_variable( y, korder, result[v], error[v], yscal );
		}
	
		double gbs_error = DBL_MAX;
		int index=-1;

		double gbs_tolerance = ComputationToDoList.ComputationalTask[SubStepDivision[0]].ThisR->gbs_tolerance;

		for(int i=0;i<korder-1;i++){
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
		//printf("%d %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n",flag,maxerror[0],maxerror[1],maxerror[2],maxerror[3],maxerror[4],maxerror[5],maxerror[6],gbs_error,H);
                //fflush(stdout);

		if( gbs_error <= gbs_tolerance ){

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

			//exit(0);
		}

		if(ThisTask==0) fflush(stdout);

		for(int i=0;i<NumDynVariables;i++){
                        free(error[i]);			total_memory -= (korder-1)*sizeof(double);
                        free(result[i]);		total_memory -= (korder-1)*sizeof(double);
                }
                free(error);				total_memory -= NumDynVariables*sizeof(double);
                free(result);				total_memory -= NumDynVariables*sizeof(double);

		free(SubStepDivision);			total_memory -= korder*sizeof(int);
		free(y);				total_memory -= korder*sizeof(double);
		free(yscal);				total_memory -= korder*sizeof(double);


	}

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

	double percent = time_interval/100;
	int epoch = -1;

	// Now proceed to integration
	int not_finished = 1;
	int counter = 0;
	int redo = 1;
	int status;

	do {

		redo = 0;

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
				mst_leapfrog( &ComputationToDoList.CopyOfSingleRegion[ctask], Hstep, ComputationToDoList.ComputationalTask[ctask].NumberOfSubsteps, &flag );

#ifdef SERIAL
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

#ifdef PARALLEL
		// We need to communicate the results
		for(int ctask=0;ctask<ComputationToDoList.NumberOfComputationalTasks;ctask++){
			int root 	= ComputationToDoList.ComputationalTask[ctask].GroupToPerformTask*NumTaskPerGbsGroup;
			double *commbuf = ComputationToDoList.CopyOfSingleRegion[ctask].State;
			int count	= ComputationToDoList.CopyOfSingleRegion[ctask].NumDynVariables;
			MPI_Bcast( commbuf, count, MPI_DOUBLE, root, MPI_COMM_WORLD );
		}
#endif

		// Now preform the GBS extrapolation, check convergence and obtain next timestep
		status = extrapolate_all_variables( NumberOfSystems, KMAX );

		// Check whether we are ready
		double time = R[0].State[R[0].TimeIndex];
		if( time > time_interval ){
			not_finished = 0;
		} else {

			int old_epoch = epoch;
			epoch = (int) floor( time/percent );
			if( epoch != old_epoch ){

				PrimMST( &R[0] );				

				//wtime1 = clock();
				//double elapsed  = (double) (wtime1-wtime0)/CLOCKS_PER_SEC;
				//double estimate = elapsed/time*time_interval; 

				//if(ThisTask==0) printf("%d percent complete. Elapsed: %.15lf. Estimate: %.15lf.\n",epoch,elapsed,estimate);
				//fflush(stdout);

			}
		}

		//if( status == 1 ){
		//	compute_U( &R[0] );
		//	R[0].Hstep = R[0].U*dt;
		//}

		//printf("time %e --- dt %.10e\n",time,R[0].Hstep);

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
        S->MSTedgeList     = calloc(N-1, sizeof(int));				total_memory += (N-1)*sizeof(int);

}


void allocate_armst_structs( struct RegularizedRegion **R, int NumberOfSystems, int MaxNumPart ){

	int MaxDynamicalVariables = 6*(MaxNumPart-1)+2;
	*R = calloc(NumberOfSystems,sizeof(struct RegularizedRegion));	total_memory += NumberOfSystems*sizeof(struct RegularizedRegion);

        for(int i=0;i<NumberOfSystems;i++){
                allocate_regularized_region( R[i], MaxNumPart );
        }

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

	total_memory = 0;

	GBSTOL  = atof(argv[4]);
	MAXPART = atoi(argv[2]);
	ETA     = atof(argv[3]);
	

//antti
for(int loop=0;loop<=0;loop++){

	char fname[99];
	char str[99];

	sprintf(str, "%.4d", loop);
	sprintf(fname,"%s%s%s", "output/output_",str,".txt");

	if(ThisTask==0){
		//printf("%s %s\n",argv[1],argv[2]);
		output = fopen(fname,"w");
		fflush(stdout);
	}

	Niter = 0;
	Nredo = 0;
	MyC = 0;
	realtime = 0;
	fictitioustime=0;
	failed_steps = 0;
	ok_steps = 0;
	time_force = 0;
	sum_order = 0;

	my_barrier();

	/////////////////////////////////////////
	struct RegularizedRegion *R;
        int NumberOfSystems = 1;
        int MaxNumPart = MAXPART;

	allerrors = calloc(6*(MaxNumPart-1)*2,sizeof(double));

	allocate_armst_structs(&R,NumberOfSystems,MaxNumPart);


#if 0
	double Pkepler;
	generate_binary_ic( &R[0], 0.2, 0.002, 0.9, 1.0*loop*M_PI/180, &Pkepler );
	double IntegrationTime = 10000*Pkepler;
#endif
#if 0
	generate_pythagorean_triplet( &R[0] );
	double IntegrationTime = 4.822e-3;
#endif

	read_ic_from_file( &R[0], argv[1] );

	my_barrier();


	previous_time = clock();
        current_time = clock();
	cpu_times = calloc(5,sizeof(clock_t));

	double IntegrationTime = 1e-4;

	into_CoM_frame( &R[0] );

	// Build the MST and integrate for some time interval

	wtime0 = clock();
	PrimMST(&R[0]);

	//analysis_for_output( &R[0], &a0, &e0, &theta0, &EN0, &Lz0 );
	//analysis_for_smallest_period_around_bh( &R[0] );

	compute_total_energy( &R[0] );
	double mye0 = R[0].H;

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

	if(ThisTask==0){
		printf("%d %d %lf %e %lf %lf %lf %lf %lf %lf %e\n",MAXPART,ok_steps+failed_steps,elapsed,myerr,(1.0*failed_steps)/(ok_steps+failed_steps),tfrac[0],tfrac[1],tfrac[2],tfrac[3],ETA,GBSTOL);
		fflush(stdout);
	}

	if(ThisTask==0){
                fclose(output);
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
	free(R[0].MSTedgeList);
	free(R[0].MSTedgeAcc);

}	// end loop

#ifdef PARALLEL
	MPI_Finalize();
#endif

	return 0;
}
