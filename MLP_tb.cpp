#include "define.h"
#include "/home/hung52852/MLP_unit/scheduled_weight_par_8_block/schedule_weight0.h"
#include "/home/hung52852/MLP_unit/scheduled_weight_par_8_block/schedule_weight1.h"
#include "/home/hung52852/MLP_unit/scheduled_weight_par_8_block/schedule_weight2.h"
#include "/home/hung52852/MLP_unit/scheduled_weight_par_8_block/schedule_weight3.h"
#include "/home/hung52852/MLP_unit/scheduled_weight_par_8_block/schedule_weight4.h"
#include "/home/hung52852/MLP_unit/scheduled_weight_par_8_block/schedule_weight5.h"
#include "/home/hung52852/MLP_unit/scheduled_weight_par_8_block/schedule_weight6.h"
#include "/home/hung52852/MLP_unit/scheduled_weight_par_8_block/schedule_weight7.h"
#include "/home/hung52852/MLP_unit/scheduled_weight_par_8_block/schedule_weight8.h"
#include "/home/hung52852/MLP_unit/scheduled_weight_par_8_block/schedule_weight9.h"
using namespace std;
void print_matrix(const output_t to_print[4][48]){
    for(int i = 0; i < 4 ; i++){
        for(int j = 0 ;j < 48 ; j++){
            cout << to_print[i][j] <<" ";
        }
        cout<<"\n";
    }
}
void ReLU(input_t data[MLP_HIDDEN_DIM], input_t res[MLP_HIDDEN_DIM]){
    for(int i = 0; i < MLP_HIDDEN_DIM; i++){
        if(data[i] < 0){
            res[i] = 0;
        }
        else{
            res[i] =  data[i];
        }
    }
}
void layer_norm(input_t data[OUT_ROW_DIM][OUT_COL_DIM], input_t res[OUT_ROW_DIM][OUT_COL_DIM], int code){

        value_t sum[OUT_ROW_DIM];
        input_t mean[OUT_ROW_DIM];

        squ_sum_t squ_sum[OUT_ROW_DIM];
        input_t squ_tmp[OUT_ROW_DIM];
        input_t std[OUT_ROW_DIM];
        for(int i = 0; i < OUT_ROW_DIM; i++){
        	sum[i] = 0;
        	squ_sum[i] = 0;
        }
        for(int i = 0; i < OUT_ROW_DIM; i++){
            for(int j = 0; j < OUT_COL_DIM; j++){
                sum[i] += data[i][j];
                squ_sum[i] += data[i][j] * data[i][j];
            }
        }    
        for(int i = 0; i < OUT_ROW_DIM; i++){
            mean[i] = sum[i] / 48;
            if(code == 1){
                cout << i <<"th golden sum = "<<sum[i]<<"\n";
                cout << i <<"th golden mean = " <<mean[i] << "\n";
            }
           
        }
        for(int i = 0; i < OUT_ROW_DIM; i++){
            squ_tmp[i] = squ_sum[i] - mean[i] * mean[i];
            std[i] = hls::sqrt(squ_tmp[i]);
            if(code == 1)
                cout << i <<"th golden std = "<< std[i]<<"\n";
        }
        
        for(int i = 0; i < OUT_ROW_DIM; i++){
            for(int ii=0; ii < MLP_HIDDEN_DIM; ii++) {
                if(std[i]!=0){

                    res[i][ii] = (data[i][ii] - mean[i]) / std[i];
                }
                else{
                    res[i][ii] = 0;
                }
            }
        }
       
}
int Checkans(output_t golden[OUT_ROW_DIM][OUT_COL_DIM], output_t out[OUT_ROW_DIM][OUT_COL_DIM]){
    for(int i = 0 ;i < OUT_ROW_DIM; i++){
        for(int j = 0; j < OUT_COL_DIM; j ++){
            if(golden[i][j] != out[i][j]){
                cout << i << "row" << j << "col \n";
                cout << "golden_matrix\n==================================================\n";
                print_matrix(golden);
                cout << "output\n ========================================================\n";
                print_matrix(out);
                return 1;
            }
        }
    }
    return 0;
}
int main(){
    ifstream infile, infile2;
    infile.open("/home/hung52852/MLP_unit/A_matrix.txt");
    infile2.open("/home/hung52852/MLP_unit/matrix.txt");
    if(!infile ){
        cout <<"open file fail!!";
        return 1;
    }
    input_t A_matrix[4][12];
    value_t B_matrix[10][12][48];
    output_t spmm_out[10][4][48];
    output_t layer_norm_out[10][4][48];
    output_t golden_matrix[10][4][48];
    output_t out_matrix[10][4][48];

    weight_t schedule_weight[10][PE_NUM][46];
    ap_uint<6> max_num[10][PE_NUM];
    for(int i = 0; i < 10 ; i++){
        for(int j = 0; j < PE_NUM; j++){
            switch(i){
                case 0:{
                    max_num[i][j] = max0[j];
                    for(int k = 0; k < 46; k++){
                        schedule_weight[i][j][k] = scheduled_weight0[j][k];
                    }
                    break;
                }
                case 1:{
                    max_num[i][j] = max1[j];
                    for(int k = 0; k < 46; k++){
                        schedule_weight[i][j][k] = scheduled_weight1[j][k];
                    }
                    break;
                }
                case 2:{
                    max_num[i][j] = max2[j];
                    for(int k = 0; k < 46; k++){
                        schedule_weight[i][j][k] = scheduled_weight2[j][k];
                    }
                    break;
                }
                case 3:{
                    max_num[i][j] = max3[j];
                    for(int k = 0; k < 46; k++){
                        schedule_weight[i][j][k] = scheduled_weight3[j][k];
                    }
                    break;
                }
                case 4:{
                    max_num[i][j] = max4[j];
                    for(int k = 0; k < 46; k++){
                        schedule_weight[i][j][k] = scheduled_weight4[j][k];
                    }
                    break;
                }
                case 5:{
                    max_num[i][j] = max5[j];
                    for(int k = 0; k < 46; k++){
                        schedule_weight[i][j][k] = scheduled_weight5[j][k];
                    }
                    break;
                }
                case 6:{
                    max_num[i][j] = max6[j];
                    for(int k = 0; k < 46; k++){
                        schedule_weight[i][j][k] = scheduled_weight6[j][k];
                    }
                    break;
                }
                case 7:{
                    max_num[i][j] = max7[j];
                    for(int k = 0; k < 46; k++){
                        schedule_weight[i][j][k] = scheduled_weight7[j][k];
                    }
                    break;
                }
                case 8:{
                    max_num[i][j] = max8[j];
                    for(int k = 0; k < 46; k++){
                        schedule_weight[i][j][k] = scheduled_weight8[j][k];
                    }
                    break;
                }
                case 9:{
                    max_num[i][j] = max9[j];
                    for(int k = 0; k < 46; k++){
                        schedule_weight[i][j][k] = scheduled_weight9[j][k];
                    }
                    break;
                }
        
            }
        }
    }
    for(int i = 0; i < 4 ; i ++){
        for(int j = 0 ; j < 12 ; j++){
            infile >> A_matrix[i][j];
        }
    }
    for(int i = 0; i < 10 ; i++){
        for(int j = 0; j < 12; j++){
            for(int k = 0 ; k < 48; k++){
                infile2 >> B_matrix[i][j][k];
            }
        }
        for(int j = 0; j < 4 ; j++){
            for(int k = 0; k < 48 ; k++){
                spmm_out[i][j][k] = 0;
            }
        }
    }
	//*********************************
    // calculate matrix multiplication
	//*********************************
    for(int i = 0; i < 10 ; i++){
        for(int j = 0; j < 4 ;j ++){
            for(int k = 0 ; k < 48 ; k++){
                for(int a = 0; a < 12 ; a++){
                    spmm_out[i][j][k] += A_matrix[j][a] * B_matrix[i][a][k];
                }
            }
        }
    }
    /*****************************
     layer norm & ReLU
     *****************************/
    for(int i = 0; i < 10 ;i++){
        layer_norm(spmm_out[i], layer_norm_out[i], i);
    }
    for(int i = 0; i < 10; i++){
        for(int j = 0; j < OUT_ROW_DIM; j++){
            ReLU(layer_norm_out[i][j], golden_matrix[i][j]);
        }
    }
    int flag = 0;
    input_t T_a_matrix[10][4][12];
    for(int i = 0; i < 10 ; i++){
        for(int j = 0; j < 4 ; j++){
            for(int k = 0; k < 12 ; k++){
                T_a_matrix[i][j][k] = A_matrix[j][k];
            }
        }
    }
    top_model(T_a_matrix, schedule_weight, max_num, out_matrix);

    for(int i = 0; i < 10; i++){
        switch(i){
            case 0:{
                cout<<i<<"th outcome\n==============================================\n";
               // top_model(A_matrix, scheduled_weight0, max0,out_matrix[i]);
				cout << "=========================================================\n";
				flag += Checkans(golden_matrix[i],out_matrix[i]);
                if(flag != 0){
                    cout << i <<"th round\n";
                    return 1;
                }
                break;
            }
            case 1:{
                cout<<i<<"th outcome\n==============================================\n";
             //   top_model(A_matrix, scheduled_weight1,max1,out_matrix[i]);
                cout<<"==============================================\n";
                flag += Checkans(golden_matrix[i],out_matrix[i]);
                if(flag != 0){
                    cout << i <<"th round\n";
                    return 1;
                }
                break;
            }
            case 2:{
                cout<<i<<"th outcome\n==============================================\n";
            //    top_model(A_matrix, scheduled_weight2, max2,out_matrix[i]);
                cout<<"==============================================\n";
                flag += Checkans(golden_matrix[i],out_matrix[i]);
                if(flag != 0){
                    cout << i <<"th round\n";
                    return 1;
                }
                break;
            }
            case 3:{
                cout<<i<<"th outcome\n==============================================\n";
             //   top_model(A_matrix, scheduled_weight3, max3,out_matrix[i]);
                cout<<"==============================================\n";
                flag += Checkans(golden_matrix[i],out_matrix[i]);
                if(flag != 0){
                    cout << i <<"th round\n";
                    return 1;
                }
                break;
            }
            case 4:{
                cout<<i<<"th outcome\n==============================================\n";
            //    top_model(A_matrix, scheduled_weight4, max4,out_matrix[i]);
                cout<<"==============================================\n";
                flag += Checkans(golden_matrix[i],out_matrix[i]);
                if(flag != 0){
                    cout << i <<"th round\n";
                    return 1;
                }
                break;
            }
            case 5:{
                cout<<i<<"th outcome\n==============================================\n";
             //   top_model(A_matrix, scheduled_weight5, max5,out_matrix[i]);
                cout<<"==============================================\n";
                flag += Checkans(golden_matrix[i],out_matrix[i]);
                if(flag != 0){
                    cout << i <<"th round\n";
                    return 1;
                }
                break;
            }
            case 6:{
                cout<<i<<"th outcome\n==============================================\n";
             //   top_model(A_matrix, scheduled_weight6,max6, out_matrix[i]);
                cout<<"==============================================\n";
                flag += Checkans(golden_matrix[i],out_matrix[i]);
                if(flag != 0){
                    cout << i <<"th round\n";
                    return 1;
                }
                break;
            }
            case 7:{
                cout<<i<<"th outcome\n==============================================\n";
            //    top_model(A_matrix, scheduled_weight7, max7,out_matrix[i]);
                cout<<"==============================================\n";
                flag += Checkans(golden_matrix[i],out_matrix[i]);
                if(flag != 0){
                    cout << i <<"th round\n";
                    return 1;
                }
                break;
            }
            case 8:{
                cout<<i<<"th outcome\n==============================================\n";
             //   top_model(A_matrix, scheduled_weight8, max8,out_matrix[i]);
                cout<<"==============================================\n";
                flag += Checkans(golden_matrix[i],out_matrix[i]);
                if(flag != 0){
                    cout << i <<"th round\n";
                    return 1;
                }
                break;
            }
            case 9:{
                cout<<i<<"th outcome\n==============================================\n";
              //  top_model(A_matrix, scheduled_weight9, max9,out_matrix[i]);
                cout<<"==============================================\n";
                flag += Checkans(golden_matrix[i],out_matrix[i]);
                if(flag != 0){
                    cout << i <<"th round\n";
                    return 1;
                }
                break;
            }
        }
    }
    
    if(flag != 0){
        return 1;
    }
    else{
        return 0;
    }
	//*************************
    // check ans
	//*************************
    // for(int i = 0; i < 10 ; i++){
    //     for(int j = 0; j < 4; j++){
    //         for(int k = 0; k < 48 ; k++){
    //             if(out_matrix[i][j][k] != golden_matrix[i][j][k]){
    //                 cout << "wrong ans\n "<< i <<"th round \n";
    //                 cout << "golden_matrix\n==================================================\n";
    //                 print_matrix(golden_matrix[i]);
    //                 cout << "output\n ========================================================\n";
    //                 print_matrix(out_matrix[i]);
    //                 return 1;
    //             }
    //             else{
    //                 // cout << "golden_matrix\n==================================================\n";
    //                 // print_matrix(golden_matrix[i]);
    //                 // cout << "output\n ========================================================\n";
    //                 // print_matrix(out_matrix[i]);
    //             }   
    //         }
    //     }
    //}

    return 0;
}
