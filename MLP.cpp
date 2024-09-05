/*
*Hung 2021/9/4
*this code is implementing a MLP unit, which includes a SPMM kernel, layer norm and ReLU
*make sure not to use static ==> cause wrong hardware and COSIM fail
*/ 
#include "define.h"
#include "util/kernel.cpp"
void ReLU(const input_t data[PE_NUM][OUT_ROW_DIM][B_matrix_par_col], output_t output[OUT_ROW_DIM][OUT_COL_DIM]){
	for(int i = 0; i < OUT_ROW_DIM; i++){
		#pragma HLS unroll
		for(int j = 0; j < OUT_COL_DIM; j++){
			#pragma HLS pipeline II = 1
			if(data[j / B_matrix_par_col][i][j % B_matrix_par_col] < 0){
				output[i][j] = 0;
			}
			else{
				output[i][j] = data[j / B_matrix_par_col][i][j % B_matrix_par_col];
			}
		}
	}
	#ifndef __SYNTHESIS__
		cout << "hardware ReLU \n ============================================ \n";
		for(int i = 0; i < OUT_ROW_DIM; i++){
			for(int j = 0; j < OUT_COL_DIM;j++){
				cout << output[i][j] <<" ";
			}
			cout <<"\n";
		}
	#endif
}
template<class T>
void Reset_Layer(T data[PE_NUM + 1][OUT_ROW_DIM]){
	Reset_Layer_loop:for(int i = 0; i <= PE_NUM; i++){
		#pragma HLS unroll
		for(int j = 0; j < OUT_ROW_DIM ; j++){
			#pragma HLS pipeline II = 1
			data[i][j] = 0;
		}
	}
}
void Layer_norm(const input_t data[PE_NUM][OUT_ROW_DIM][B_matrix_par_col], output_t res[PE_NUM][OUT_ROW_DIM][B_matrix_par_col]){
	//#pragma HLS PIPELINE
	#pragma HLS INLINE OFF


	   //attr_t sum = 0;
	   // input_t sum[MLP_HIDDEN_DIM];
	   	value_t sum[PE_NUM + 1][OUT_ROW_DIM];
       	#pragma HLS array_partition variable = sum dim = 0 complete
	   	input_t mean[OUT_ROW_DIM];

	   //attr_t squ_sum = 0;
	   	squ_sum_t squ_sum[PE_NUM + 1][OUT_ROW_DIM];
       	#pragma HLS array_partition variable = squ_sum dim = 0 complete
	   	input_t squ_tmp[OUT_ROW_DIM];
	   	input_t std[OUT_ROW_DIM];
		/*
		Reset 
		*/
		Reset_Layer<value_t>(sum);
		Reset_Layer<squ_sum_t>(squ_sum);

		sum_loop:for(int ii=0; ii<PE_NUM; ii++) {
			#pragma HLS unroll
			for(int j = 0; j < OUT_ROW_DIM; j++){
				#pragma HLS pipeline II = 1
				for(int k = 0; k < B_matrix_par_col; k++){
					sum[ii][j] += data[ii][j][k];
					squ_sum[ii][j] += data[ii][j][k] * data[ii][j][k];
				}
			}
		}


		sum_loop_2:for(int ii = 0; ii < PE_NUM; ii++){
			#pragma HLS unroll
			for(int j = 0; j < OUT_ROW_DIM; j++){
				#pragma HLS pipeline II = 1
				squ_sum[PE_NUM][j] += squ_sum[ii][j];
				sum[PE_NUM][j] += sum[ii][j];
			}
		}

		mean_square_loop:for(int i = 0; i < OUT_ROW_DIM; i++){
			#pragma HLS unroll
			mean[i] = sum[PE_NUM][i] / 48; 
			// try to use BRAM
			squ_tmp[i] = squ_sum[PE_NUM][i] - mean[i] * mean[i];
			std[i] = hls::sqrt(squ_tmp[i]);
			//fxp_sqrt(std, squ_tmp);
		}
		
		

	  	cal_write_back_loop:for(int ii=0; ii<PE_NUM; ii++) {
			#pragma HLS unroll
			for(int j = 0; j < OUT_ROW_DIM; j++){
				for(int k = 0; k < B_matrix_par_col; k++){
					if(std[j]!= 0){
						res[ii][j][k] = (data[ii][j][k] - mean[j]) / std[j];
					}
					else{
						res[ii][j][k] = 0;
					}
				}
			}
				
	  	}
		#ifndef __SYNTHESIS__
			for(int i = 0; i < OUT_ROW_DIM; i++){
				cout << i << "th sum = "<<sum[PE_NUM][i]<<"\n";
				cout << i << "th mean = "<<mean[i]<<"\n";
				cout << i << "th std  = "<<std[i] <<"\n";
			}
			cout << "hardware layer norm\n ===============================================================\n";
			for(int i = 0; i < OUT_ROW_DIM; i++){
				for(int j = 0; j < PE_NUM; j ++){
					for(int k = 0; k < B_matrix_par_col; k++){
						cout << res[j][i][k] <<" ";
					}
				}
				cout <<"\n";
			}
			cout << "===========================================================================\n";
		#endif
}
void reset_outmatrix(output_t output_matrix[10][OUT_ROW_DIM][OUT_COL_DIM]){
	reset_out_loop:for(int i = 0; i < 10; i++){
		#pragma HLS unroll
		for(int j = 0; j < OUT_ROW_DIM; j++){
			for(int k = 0; k < OUT_COL_DIM; k++){
				#pragma HLS pipeline II = 1
				output_matrix[i][j][k] = 0;
			}
		}
	}
}
void top_model(const input_t A_matrix[10][IN_ROW_DIM][IN_COL_DIM],const weight_t non_zero_list[10][PE_NUM][MAX_NON_ZERO], const ap_uint<6> max[10][PE_NUM], output_t output_matrix[10][OUT_ROW_DIM][OUT_COL_DIM]){
	


	#pragma HLS array_partition variable = non_zero_list dim = 1 complete
	#pragma HLS array_partition variable = non_zero_list dim = 2 complete
	
	#pragma HLS array_partition variable = max dim = 0 complete
  
	#pragma HLS array_partition variable = output_matrix dim = 1 complete
	
	#pragma HLS array_partition variable = A_matrix dim = 1 complete
	#pragma HLS array_partition variable = A_matrix dim = 2 complete
	
	#pragma HLS INLINE off
	input_t ping_pong_spmm_out[2][PE_NUM][OUT_ROW_DIM][B_matrix_par_col];
	input_t ping_pong_layer_norm[2][PE_NUM][OUT_ROW_DIM][B_matrix_par_col];

	#pragma HLS array_partition variable = ping_pong_spmm_out dim = 1 complete
	#pragma HLS array_partition variable = ping_pong_spmm_out dim = 2 complete
	#pragma HLS array_partition variable = ping_pong_spmm_out dim = 3 complete

	#pragma HLS array_partition variable = ping_pong_layer_norm dim = 1 complete
	#pragma HLS array_partition variable = ping_pong_layer_norm dim = 2 complete

	//reset_outmatrix(output_matrix);
	// Spmm_kernel(A_matrix, non_zero_list, max, ping_pong_spmm_out[0]);
	// Layer_norm(ping_pong_spmm_out[0], ping_pong_layer_norm[0]);
	// ReLU(ping_pong_layer_norm[0], output_matrix);
	//first cycle
	Spmm_kernel(A_matrix[0], non_zero_list[0], max[0], ping_pong_spmm_out[0]);

	// second cycle
	Spmm_kernel(A_matrix[1], non_zero_list[1], max[1], ping_pong_spmm_out[1]);
	Layer_norm(ping_pong_spmm_out[0], ping_pong_layer_norm[0]);


	//NUM of test data = 10
	cycle_loop:for(int i = 3 ; i <= 10 ;i++){
		if(i % 2 == 1){
			Spmm_kernel(A_matrix[i - 1], non_zero_list[i - 1], max[i - 1], ping_pong_spmm_out[0]);
			Layer_norm(ping_pong_spmm_out[1], ping_pong_layer_norm[1]);
			ReLU(ping_pong_layer_norm[0], output_matrix[i - 3]);
		}
		else {
			Spmm_kernel(A_matrix[i - 1], non_zero_list[i - 1], max[i - 1], ping_pong_spmm_out[1]);
			Layer_norm(ping_pong_spmm_out[0], ping_pong_layer_norm[0]);
			ReLU(ping_pong_layer_norm[1], output_matrix[i - 3]);
		}
	}
	// last few round
	Layer_norm(ping_pong_spmm_out[1], ping_pong_layer_norm[1]);
	ReLU(ping_pong_layer_norm[0], output_matrix[8]);

	ReLU(ping_pong_layer_norm[1], output_matrix[9]);
}
