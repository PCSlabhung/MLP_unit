#include "../define.h"
using namespace std;
void copy_mem(const input_t A_matrix[IN_ROW_DIM][IN_COL_DIM], input_t copy[PE_NUM][IN_ROW_DIM][IN_COL_DIM]){
    for(int i = 0 ; i < IN_ROW_DIM; i++){
        #pragma HLS pipeline II = 1
        for(int j = 0 ; j < IN_COL_DIM; j++){
            #pragma HLS UNROLL
            for(int k = 0; k < PE_NUM; k++){
                #pragma HLS unroll
                copy[k][i][j] = A_matrix[i][j];
            }
        }
    }
}
void PU(const input_t& A_value, const value_t& B_value, const index_t o_index_col,output_t out[OUT_COL_DIM]){
	#pragma HLS pipeline II = 1
    output_t temp = out[o_index_col];
	value_t value = A_value * B_value;
	out[o_index_col] = temp + value;
}
void PE(const input_t A_matrix[IN_ROW_DIM][IN_COL_DIM],const weight_t non_zero_list[MAX_NON_ZERO], output_t output_matrix[OUT_ROW_DIM][OUT_COL_DIM]){
	#pragma HLS bind_storage variable = output_matrix type = RAM_S2P
	for(int i = 0 ; i < MAX_NON_ZERO ; i++){
		#pragma HLS pipeline II = 1
		// PU_level parallel
        
		for(int j = 0; j < IN_ROW_DIM; j++){
            #pragma HLS unroll factor = IN_ROW_DIM
            #pragma HLS pipeline II =1
            #pragma HLS inline off
            index_t row = non_zero_list[i].row;
            index_t col = non_zero_list[i].col;
            value_t value = non_zero_list[i].value;
            PU(A_matrix[j][row], value, col,output_matrix[j]);
           
        }
	}
}
void PU_v1(const input_t& A_value, const value_t& B_value, const index_t o_index_col,output_t out[OUT_COL_DIM / PE_NUM]){
	#pragma HLS pipeline II = 1
    output_t temp = out[o_index_col];
	value_t value = A_value * B_value;
	out[o_index_col] = temp + value;
}
void PE_v1(const input_t A_matrix[IN_ROW_DIM][IN_COL_DIM],const weight_t non_zero_list[MAX_NON_ZERO], const ap_uint<6> max, output_t output_matrix[OUT_ROW_DIM][OUT_COL_DIM / PE_NUM]){
	#pragma HLS bind_storage variable = output_matrix type = RAM_S2P
	non_zero_iteration_loop:for(int i = 0 ; i < MAX_NON_ZERO ; i++){
		#pragma HLS pipeline II = 1
		// PU_level parallel
        if(i == max)
            break;
		PU_loop:for(int j = 0; j < IN_ROW_DIM; j++){
            #pragma HLS unroll factor = IN_ROW_DIM
            #pragma HLS pipeline II =1
            #pragma HLS inline off
            index_t row = non_zero_list[i].row;
            index_t col = non_zero_list[i].col % (B_matrix_par_col);
            value_t value = non_zero_list[i].value;
            PU_v1(A_matrix[j][row], value, col,output_matrix[j]);
            // if(value != 0){
            //     cout << "row = "<<j <<"original col = "<<non_zero_list[i].col <<"col = "<<col<<"value = "<< output_matrix[j][col]<<"\n";
            // }
                
        }
	}
}
void reset_output(output_t matrix[OUT_ROW_DIM][OUT_COL_DIM / PE_NUM]){
    Reset_outer_loop:for(int i = 0; i < OUT_ROW_DIM; i++){
        Reset_inner_loop:for(int j = 0; j < OUT_COL_DIM / PE_NUM;  j++){
            matrix[i][j] = 0;
        }
    }
}
void move_mem(const output_t in[PE_NUM][OUT_ROW_DIM][OUT_COL_DIM / PE_NUM], output_t output[OUT_ROW_DIM][OUT_COL_DIM]){
    for(int j = 0; j < OUT_COL_DIM; j++){
        //#pragma HLS pipeline II = 1
        for(int i = 0; i < OUT_ROW_DIM; i++){
            #pragma HLS unroll
            output[i][j] = in[j / (B_matrix_par_col)][i][j % (B_matrix_par_col)];
        }
    }
}
void Spmm_kernel(const input_t A_matrix[IN_ROW_DIM][IN_COL_DIM],const weight_t non_zero_list[PE_NUM][MAX_NON_ZERO], const ap_uint<6> max[PE_NUM],output_t output_matrix[PE_NUM][OUT_ROW_DIM][B_matrix_par_col]){
    input_t copy_memory[PE_NUM][IN_ROW_DIM][IN_COL_DIM];
    
    //output_t hand_partition_output[PE_NUM][OUT_ROW_DIM][OUT_COL_DIM / PE_NUM];
    #pragma HLS array_partition variable = A_matrix dim = 1 complete
    #pragma HLS array_partition variable = copy_memory dim = 1 complete
    #pragma HLS array_partition variable = copy_memory dim = 2 complete

    // #pragma HLS INTERFACE ap_memory port = output_matrix storage_type=RAM_S2P
    // #pragma HLS array_partition variable = output_matrix dim = 2 type = block factor = PE_NUM
  //  #pragma HLS bind_storage variable = output_matrix type = RAM_S2p
    #pragma HLS array_partition variable = max dim = 0 complete
    
    #pragma HLS array_partition variable = non_zero_list dim = 1 complete
    /*
    change hand_partition_output to output matrix
    use ping pong buffer to deal with memory port issue
    */
    #pragma HLS array_partition variable = output_matrix dim = 1 complete
    #pragma HLS array_partition variable = output_matrix dim = 2 complete // new try
//    #pragma HLS array_partition variable = hand_partition_output dim = 3 factor = PE_NUM // make things worst
    #pragma HLS bind_storage variable = output_matrix type = RAM_S2P // or RAM_2P
    #pragma HLS inline off    
    copy_mem(A_matrix, copy_memory);
    reset_LOOP:for(int i = 0; i < PE_NUM; i++){
        #pragma HLS unroll
        reset_output(output_matrix[i]);
    }

    PE_LOOP:for(int i = 0; i < PE_NUM ; i++){
        #pragma HLS unroll 
        //#pragma HLS inline off
        //PE(copy_memory[i], non_zero_list[i], output_matrix);
        PE_v1(copy_memory[i], non_zero_list[i], max[i],output_matrix[i]);
    }
    #ifndef __SYNTHESIS__
        cout <<"hardware spmm out \n ================================================\n";
        for(int i = 0; i < OUT_ROW_DIM; i++){
            for(int j = 0; j < PE_NUM; j ++){
                for(int k = 0; k < B_matrix_par_col; k++){
                    cout << output_matrix[j][i][k] << " ";
                }
            }
            cout << "\n";
        }
        cout << "==================================================\n";
    #endif
    //move_mem(hand_partition_output, output_matrix);   
}