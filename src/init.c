// Functions for initialization of the integrator


#if 0
// TODO leave only the parallel options
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
#if defined(PRIM) && defined(PRIM_DIVIDE_CONQUER)
        printf("Please select EITHER prim OR prim_divide_conquer. Currently both selected.\n");
	exit(0);
#endif
#if !defined(PRIM) && !defined(PRIM_DIVIDE_CONQUER)
	printf("Please select EITHER prim OR prim_divide_conquer. Currently none selected.\n");
	exit(0);
#endif

// Init MPI if needed
#ifdef PARALLEL
	MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &Ntask);
        MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);

	NumGbsGroup        = 4;
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
        S->MSTedgeAcc   = calloc(3*(N-1), sizeof(double));

        S->AccPN          = calloc(3*N,     sizeof(double));
        S->AuxVel         = calloc(3*N,     sizeof(double));
        S->AuxEdgeVel     = calloc(3*(N-1), sizeof(double));
        S->MSTedgeAcc_PN  = calloc(3*(N-1), sizeof(double));

        S->Vertex          = calloc(N,          sizeof(struct GraphVertex));

	double safetybuf = 1.05;
	int NumLocalEdge = (int) floor( safetybuf*S->NumEdge/Ntask );
	S->LocalEdge       = calloc(NumLocalEdge, sizeof(struct GraphEdge));
	S->LocalEdgeSubset = calloc(NumLocalEdge, sizeof(struct GraphEdge));
	S->EdgeInMST       = calloc(N-1, sizeof(struct GraphEdge));

}

void allocate_armst_structs( struct RegularizedRegion **R, int MaxNumPart ){

        int MaxDynamicalVariables = 6*(MaxNumPart-1)+2;
        *R = calloc(1,sizeof(struct RegularizedRegion));

        allocate_regularized_region( R[0], MaxNumPart );

        gbsS = calloc(KMAX,sizeof(double *));
        for(int i=0;i<KMAX;i++){
		gbsS[i] = calloc(KMAX,sizeof(double));
        }

       int korder = 8;
       int NumDynVariables = MaxDynamicalVariables;

        errorbuf  = calloc(NumDynVariables,sizeof(double));
        resultbuf = calloc(NumDynVariables,sizeof(double));
        y = calloc(korder,sizeof(double));
	yscal = calloc(korder,sizeof(double));
        maxerror = calloc(korder-1,sizeof(double));
        error = calloc(NumDynVariables,sizeof(double *));
        result = calloc(NumDynVariables,sizeof(double *));

        for(int i=0;i<NumDynVariables;i++){
	        error[i] = calloc(korder-1,sizeof(double));
                result[i] = calloc(korder-1,sizeof(double));
        }
        indexlist = calloc(8,sizeof(int *));

        for(int i=0;i<8;i++){
          indexlist[i] = calloc(MaxNumPart, sizeof(int));
	}


	ParticleInNode = calloc(MaxNumPart,sizeof(struct OctreeNode *));
        int NumAlloc = KMAX;

        ComputationToDoList.ComputationalTask = calloc(NumAlloc,sizeof(struct ComTask));
        for(int i=0;i<NumAlloc;i++){
                ComputationToDoList.ComputationalTask[i].ResultState = calloc(MaxDynamicalVariables, sizeof(double));
        }
        ComputationToDoList.CopyOfSingleRegion = calloc(NumAlloc,sizeof(struct RegularizedRegion));
        for(int i=0;i<NumAlloc;i++){
                allocate_regularized_region( &ComputationToDoList.CopyOfSingleRegion[i], MaxNumPart );
        }
}

// free
void free_data( struct RegularizedRegion *R ){

        for(int i=0;i<KMAX;i++){
		free(ComputationToDoList.ComputationalTask[i].ResultState);
        }
        free(ComputationToDoList.ComputationalTask);

        free(R->LocalEdge);
        free(R->Vertex);
        free(R->Pos);
	free(R->Vel);
        free(R->Acc);
        free(R->Mass);
        free(R->State);
        free(R->MSTedgeAcc);
        free(R->EdgeInMST);
	free(R);

	free(gbsS);

}
#endif
