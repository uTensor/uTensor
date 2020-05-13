
#ifndef constants_quantized_fully_connected_hpp 
#define constants_quantized_fully_connected_hpp 

 
static const int8_t s_ref_A[2] = { 
 -2,  0  
};
static const int32_t s_ref_A_zp [1] = { -128 };
static const float s_ref_A_scale [1] = { 0.13013659417629242 };

 
static const int8_t s_ref_filter[4] = { 
 -56,  -71,  76,  127  
};
static const int32_t s_ref_filter_zp [1] = { 0 };
static const float s_ref_filter_scale [1] = { 0.009406298398971558 };

 
static const int32_t s_ref_bias[2] = { 
 0,  -4  
};
static const int32_t s_ref_bias_zp [1] = { 0 };
static const float s_ref_bias_scale [1] = { 0.0012241036165505648 };

 
static const int8_t s_ref_output_ref[2] = { 
 -126,  -113  
};
static const int32_t s_ref_output_ref_zp [1] = { -128 };
static const float s_ref_output_ref_scale [1] = { 0.24408094584941864 };


#endif