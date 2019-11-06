#ifndef __UTENSOR_TYPES_H
#define __UTENSOR_TYPES_H
#include <cstdint>

class TensorShape {
        TensorShape(uint16_t shape);
        TensorShape(uint16_t shape[1]) ;
        TensorShape(uint16_t shape[2]) ;
        TensorShape(uint16_t shape[3]) ;
        TensorShape(uint16_t shape[4]) ;

        uint16_t operator[] (int i) const ;
        uint16_t& operator[](int i) ;
        void update_dims();
        uint16_t get_linear_size() const; 
    private:
        uint16_t _shape[4];
        uint8_t  _num_dims;
}
// Do something to remember current type
enum ttype: uint8_t {
            i8,
            u8,
            i16,
            u16,
            i32,
            u32,
            float,
            undefined
};

// Need to figure out way of defering reference until after lefthand assignment
class IntegralValue {
    void* p;
    public:

        // Explicit
        IntegralValue(void* p) : p(p) ;
        IntegralValue(const uint8_t& u) ;
        IntegralValue(const int8_t& u);
        IntegralValue(const uint16_t& u);
        IntegralValue(const int16_t& u);
        IntegralValue(const uint32_t& u);
        IntegralValue(const int32_t& u);

        //IntegralValue& operator=(void* _p) { p = _p; }

        operator uint8_t   ( ) const; 
        operator uint8_t&  ( )      ; 
        operator int8_t    ( ) const; 
        operator int8_t&   ( )      ; 
        
        operator uint16_t  ( ) const; 
        operator uint16_t& ( )      ; 
        operator int16_t   ( ) const; 
        operator int16_t&  ( )      ; 

        operator uint32_t  ( ) const; 
        operator uint32_t& ( )      ; 
        operator int32_t   ( ) const; 
        operator int32_t&  ( )      ; 

        operator float   ( ) const  ;
        operator float&  ( )        ;

};

#endif
